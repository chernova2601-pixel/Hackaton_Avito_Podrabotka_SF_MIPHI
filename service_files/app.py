from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from hackaton.service.dto import (
    BatchEventsRequest,
    BatchShiftsRequest,
    BatchUsersRequest,
    PredictRequest,
)
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests", ["endpoint"])
LOGGER = logging.getLogger(__name__)


class HackatonRpcService:
    def __init__(self, repository: Repository, prepare: PrepareManager) -> None:
        self.repository = repository
        self.prepare_manager = prepare

        self.models = None  # список моделей LightGBM (ансамбль)
        self.feature_cols = None
        self.shift_features = None  # dict shift_id -> признаки смены
        self.user_stats = None  # dict user_id -> статистики пользователя
        self.default_stats = None
        self.model_loaded = False
        self.model_loading = False

    async def _load_model(self):
        """Фоновая загрузка артефактов модели."""
        self.model_loading = True
        try:
            ARTIFACT_DIR = Path("artifacts/train")
            self.models = joblib.load(ARTIFACT_DIR / "model.pkl")
            self.feature_cols = joblib.load(ARTIFACT_DIR / "feature_columns.pkl")

            shift_df = joblib.load(ARTIFACT_DIR / "shift_features.pkl")
            self.shift_features = shift_df.set_index("shift_id").to_dict("index")

            user_stats_df = joblib.load(ARTIFACT_DIR / "user_stats.pkl")
            self.user_stats = user_stats_df.set_index("user_id").to_dict("index")
            self.default_stats = joblib.load(ARTIFACT_DIR / "default_user_stats.pkl")

            self.model_loaded = True
            LOGGER.info("Model artifacts loaded successfully")
        except Exception as e:
            LOGGER.error(f"Failed to load model artifacts: {e}")
            self.model_loaded = False
        finally:
            self.model_loading = False

    async def user(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("user").inc()
        with REQUEST_LATENCY.labels("user").time():
            request = BatchUsersRequest.model_validate(payload)
            LOGGER.info("RPC user called, batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_users(request.items)
            return {"accepted": accepted}

    async def user_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("user_stat").inc()
        with REQUEST_LATENCY.labels("user_stat").time():
            LOGGER.info("RPC user_stat called")
            return {"count": await self.repository.count_table("users")}

    async def event(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("event").inc()
        with REQUEST_LATENCY.labels("event").time():
            request = BatchEventsRequest.model_validate(payload)
            LOGGER.info("RPC event called, batch_size=%s", len(request.items))
            accepted = await self.repository.insert_events(request.items)
            return {"accepted": accepted}

    async def event_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("event_stat").inc()
        with REQUEST_LATENCY.labels("event_stat").time():
            LOGGER.info("RPC event_stat called")
            return {"count": await self.repository.count_table("events")}

    async def shift(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("shift").inc()
        with REQUEST_LATENCY.labels("shift").time():
            request = BatchShiftsRequest.model_validate(payload)
            LOGGER.info("RPC shift called, batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_shifts(request.items)
            return {"accepted": accepted}

    async def shift_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("shift_stat").inc()
        with REQUEST_LATENCY.labels("shift_stat").time():
            LOGGER.info("RPC shift_stat called")
            return {"count": await self.repository.count_table("shifts")}

    async def prepare(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("prepare").inc()
        with REQUEST_LATENCY.labels("prepare").time():
            LOGGER.info("RPC prepare called")
            started = await self.prepare_manager.start()
            if not started:
                return {"status": "already_running", "status_code": 409}
            asyncio.create_task(self._load_model())
            return {"status": "started", "status_code": 200}

    async def ready(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("ready").inc()
        with REQUEST_LATENCY.labels("ready").time():
            LOGGER.info("RPC ready called")
            if not self.prepare_manager.ready or not self.model_loaded:
                return {"ready": False, "status_code": 425}
            return {"ready": True, "status_code": 200}

    async def predict(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("predict").inc()
        with REQUEST_LATENCY.labels("predict").time():
            LOGGER.info("RPC predict called")

            # Если prepare_manager ещё не готов или модель загружается – возвращаем 503
            if not self.prepare_manager.ready or self.model_loading:
                return {"user_ids": [], "status_code": 503, "detail": "model is in prepare state"}

            # Если prepare_manager готов, но модель не загружена – fallback
            if not self.model_loaded:
                try:
                    request = PredictRequest.model_validate(payload)
                except ValidationError as exc:
                    return {"user_ids": [], "status_code": 422, "detail": str(exc)}
                candidates = await self.repository.find_top_candidates(
                    location_id=request.shift.location_id,
                    need_mk=request.shift.need_mk,
                    limit=request.limit,
                )
                if not candidates:
                    candidates = await self.repository.fallback_candidates(limit=request.limit)
                if not candidates:
                    return {"user_ids": [], "status_code": 400, "detail": "no users loaded"}
                return {"user_ids": candidates[: request.limit], "status_code": 200}

            # Основная логика с моделью
            try:
                request = PredictRequest.model_validate(payload)
            except ValidationError as exc:
                return {"user_ids": [], "status_code": 422, "detail": str(exc)}

            shift_id = request.shift.id
            candidates = await self.repository.find_top_candidates(
                location_id=request.shift.location_id,
                need_mk=request.shift.need_mk,
                limit=request.limit * 2,
            )
            if not candidates:
                candidates = await self.repository.fallback_candidates(limit=request.limit)
            if not candidates:
                return {"user_ids": [], "status_code": 400, "detail": "no users loaded"}

            s_feat = self.shift_features.get(shift_id)
            if s_feat is None:
                return {"user_ids": candidates[: request.limit], "status_code": 200}

            features_list = []
            valid_candidates = []
            for uid in candidates:
                u_stat = self.user_stats.get(uid, self.default_stats)

                feat = {
                    "user_hist_views": u_stat.get("user_hist_views", 0),
                    "user_hist_applies": u_stat.get("user_hist_applies", 0),
                    "user_hist_finished": u_stat.get("user_hist_finished", 0),
                    "user_hist_cancels": u_stat.get("user_hist_cancels", 0),
                    "hist_view_cnt": 0,
                    "hist_apply_cnt": 0,
                    "hist_finished_cnt": 0,
                    "hist_user_cancel_cnt": 0,
                    "hist_system_cancel_cnt": 0,
                    "shift_popularity": s_feat.get("shift_popularity", 0),
                    "employer_popularity": s_feat.get("employer_popularity", 0),
                    "location_popularity": s_feat.get("location_popularity", 0),
                    "task_popularity": s_feat.get("task_popularity", 0),
                    "shift_hour": pd.to_datetime(s_feat["start_at"]).hour,
                    "shift_weekday": pd.to_datetime(s_feat["start_at"]).weekday,
                    "shift_is_weekend": int(pd.to_datetime(s_feat["start_at"]).weekday >= 5),
                    "hour_sin": np.sin(2 * np.pi * pd.to_datetime(s_feat["start_at"]).hour / 24),
                    "hour_cos": np.cos(2 * np.pi * pd.to_datetime(s_feat["start_at"]).hour / 24),
                    "weekday_sin": np.sin(
                        2 * np.pi * pd.to_datetime(s_feat["start_at"]).weekday / 7
                    ),
                    "weekday_cos": np.cos(
                        2 * np.pi * pd.to_datetime(s_feat["start_at"]).weekday / 7
                    ),
                    "reward": s_feat["reward"],
                    "hours": s_feat["hours"],
                    "capacity": s_feat["capacity"],
                    "reward_per_hour": s_feat["reward"] / (s_feat["hours"] + 1e-5),
                    "reward_log": np.log1p(s_feat["reward"]),
                    "capacity_log": np.log1p(s_feat["capacity"]),
                    "capacity_ratio_to_10": s_feat["capacity"] / 10.0,
                    "needed_candidates": min(s_feat["capacity"], 10),
                    "location_match": int(
                        s_feat.get("shift_location", "") == u_stat.get("user_location", "")
                    ),
                    "mk_match": int(s_feat.get("need_mk", False) == u_stat.get("has_mk", False)),
                    "has_mk": int(u_stat.get("has_mk", False)),
                    "is_strict_location": int(u_stat.get("is_strict_location", False)),
                    "need_mk": int(s_feat.get("need_mk", False)),
                    "id_differential": int(s_feat.get("id_differential", False)),
                    "user_avg_reward": u_stat.get("user_avg_reward", 0),
                    "user_avg_hours": u_stat.get("user_avg_hours", 0),
                    "user_avg_shift_hour": u_stat.get("user_avg_shift_hour", 0),
                    "user_avg_shift_weekday": u_stat.get("user_avg_shift_weekday", 0),
                    "user_pref_event_count": u_stat.get("user_pref_event_count", 0),
                    "reward_preference_match": u_stat.get("reward_preference_match", 0),
                    "hours_preference_match": u_stat.get("hours_preference_match", 0),
                    "hour_preference_match": u_stat.get("hour_preference_match", 0),
                    "weekday_preference_match": u_stat.get("weekday_preference_match", 0),
                    }
                features_list.append(feat)
                valid_candidates.append(uid)

            if not features_list:
                return {"user_ids": candidates[: request.limit], "status_code": 200}

            X = pd.DataFrame(features_list)[self.feature_cols]
            probas = [model.predict_proba(X)[:, 1] for model in self.models]
            probs = np.mean(probas, axis=0)

            top_idx = np.argsort(probs)[-10:][::-1]
            top_users = [valid_candidates[i] for i in top_idx]

            return {"user_ids": top_users, "status_code": 200}

    async def health(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("health").inc()
        LOGGER.info("RPC health called")
        return {"status": "ok", "status_code": 200}

    async def metrics(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("metrics").inc()
        LOGGER.info("RPC metrics called")
        return {
            "content_type": CONTENT_TYPE_LATEST,
            "payload": generate_latest().decode("utf-8"),
            "status_code": 200,
        }
