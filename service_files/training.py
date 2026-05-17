from __future__ import annotations

import json
import logging
import pickle
import joblib
from dataclasses import asdict, dataclass
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

from hackaton.eval.metric import calculate_target_metric

LOGGER = logging.getLogger(__name__)

REQUIRED_USER_COLUMNS = ["location_id", "is_strict_location", "id", "has_mk"]
REQUIRED_SHIFT_COLUMNS = [
    "id",
    "start_at",
    "location_id",
    "task_type",
    "employer_id",
    "workplace_id",
    "need_mk",
    "id_differential",
    "hours",
    "reward",
    "capacity"]
REQUIRED_EVENT_COLUMNS = ["id", "shift_id", "user_id", "interaction", "ts"]
VALID_INTERACTIONS = {"VIEW", "APPLY", "FINISHED", "USER_CANCEL", "SYSTEM_CANCEL"}

@dataclass(frozen=True, slots=True)
class TrainConfig:
    user_path: str
    shift_path: str
    event_path: str
    output_dir: str
    random_state: int = 42
    max_iter: int = 1000
    test_ratio: float = 0.2
    skip_shap: bool = True
    shap_sample_size: int = 1000
    
def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping)

def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> dict:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    return {"rows": int(len(df)), "required_columns_ok": True}

def _load_and_validate_data(
    cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    users = pd.read_csv(cfg.user_path)
    shifts = pd.read_csv(cfg.shift_path)
    events = pd.read_csv(cfg.event_path)

    checks = {
        "user": _validate_columns(users, REQUIRED_USER_COLUMNS, "user.csv"),
        "shift": _validate_columns(shifts, REQUIRED_SHIFT_COLUMNS, "shift.csv"),
        "event": _validate_columns(events, REQUIRED_EVENT_COLUMNS, "event.csv"),
    }

    users = users[REQUIRED_USER_COLUMNS].copy()
    shifts = shifts[REQUIRED_SHIFT_COLUMNS].copy()
    events = events[REQUIRED_EVENT_COLUMNS].copy()

    users["id"] = users["id"].astype(str)
    users["location_id"] = users["location_id"].astype(str)
    users["has_mk"] = _to_bool(users["has_mk"])
    users["is_strict_location"] = _to_bool(users["is_strict_location"])

    shifts["id"] = shifts["id"].astype(str)
    shifts["location_id"] = shifts["location_id"].astype(str)
    shifts["task_type"] = shifts["task_type"].astype(str)
    shifts["employer_id"] = shifts["employer_id"].astype(str)
    shifts["workplace_id"] = shifts["workplace_id"].astype(str)
    shifts["need_mk"] = _to_bool(shifts["need_mk"])
    shifts["id_differential"] = _to_bool(shifts["id_differential"])
    shifts["hours"] = pd.to_numeric(shifts["hours"], errors="coerce")
    shifts["reward"] = pd.to_numeric(shifts["reward"], errors="coerce")
    shifts["capacity"] = pd.to_numeric(shifts["capacity"], errors="coerce")
    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")

    events["id"] = events["id"].astype(str)
    events["shift_id"] = events["shift_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)
    events["interaction"] = events["interaction"].astype(str).str.upper()
    events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")
    events = events[events["interaction"].isin(VALID_INTERACTIONS)]

    users = users.drop_duplicates(subset=["id"]).dropna(subset=["id", "location_id"])
    shifts = shifts.drop_duplicates(subset=["id"]).dropna(subset=["id", "start_at", "location_id"])
    events = events.drop_duplicates(subset=["id"]).dropna(subset=["shift_id", "user_id", "ts"])

    checks["post_clean_rows"] = {
        "user": len(users),
        "shift": len(shifts),
        "event": len(events),
    }
    return users, shifts, events, checks
#строим признаки для обучения
def _build_features(users: pd.DataFrame, shifts: pd.DataFrame, history_events: pd.DataFrame, target_pairs: pd.DataFrame) -> pd.DataFrame:
    shifts_renamed = shifts.rename(columns={"id": "shift_id"})

    # - Исторические агрегаты на history_events -
    merged_hist = history_events.merge(shifts_renamed, on="shift_id", how="inner")
    merged_hist = merged_hist[merged_hist["ts"] <= merged_hist["start_at"]].copy()

    # Агрегаты по паре (исторические)
    hist_pair = merged_hist.groupby(["user_id", "shift_id"], as_index=False).agg(
        hist_view_cnt=("interaction", lambda s: (s == "VIEW").sum()),
        hist_apply_cnt=("interaction", lambda s: (s == "APPLY").sum()),
        hist_finished_cnt=("interaction", lambda s: (s == "FINISHED").sum()),
        hist_user_cancel_cnt=("interaction", lambda s: (s == "USER_CANCEL").sum()),
        hist_system_cancel_cnt=("interaction", lambda s: (s == "SYSTEM_CANCEL").sum()))

    # Агрегаты пользователя (исторические)
    user_hist = merged_hist.groupby("user_id", as_index=False).agg(
        user_hist_views=("interaction", lambda s: (s == "VIEW").sum()),
        user_hist_applies=("interaction", lambda s: (s == "APPLY").sum()),
        user_hist_finished=("interaction", lambda s: (s == "FINISHED").sum()),
        user_hist_cancels=("interaction", lambda s: (s == "USER_CANCEL").sum()))

    # -Популярности (только на истории) -
    shift_pop = merged_hist[merged_hist["interaction"] == "APPLY"].groupby("shift_id").size().reset_index(name="shift_popularity")
    employer_pop = merged_hist[merged_hist["interaction"] == "APPLY"].groupby("employer_id").size().reset_index(name="employer_popularity")
    location_pop = merged_hist[merged_hist["interaction"] == "APPLY"].groupby("location_id").size().reset_index(name="location_popularity")
    task_pop = merged_hist[merged_hist["interaction"] == "APPLY"].groupby("task_type").size().reset_index(name="task_popularity")

    # - Объединяем с target_pairs -
    result = target_pairs.merge(hist_pair, on=["user_id", "shift_id"], how="left")
    result = result.merge(user_hist, on="user_id", how="left")
    result = result.merge(shifts_renamed, on="shift_id", how="left")
    result = result.merge(users, left_on="user_id", right_on="id", how="left", suffixes=("", "_user"))
    result = result.merge(shift_pop, on="shift_id", how="left")
    result = result.merge(employer_pop, on="employer_id", how="left")
    result = result.merge(location_pop, left_on="location_id", right_on="location_id", how="left")
    result = result.merge(task_pop, on="task_type", how="left")

    # Убираем из пользовательских агрегатов события текущей пары (user, shift)
    for col in ['hist_view_cnt', 'hist_apply_cnt', 'hist_finished_cnt', 'hist_user_cancel_cnt', 'hist_system_cancel_cnt']:
        if col in result.columns:
            user_col = col.replace('hist_', 'user_hist_')
            if user_col in result.columns:
                result[user_col] = (result[user_col] - result[col]).clip(lower=0)

    if 'hist_apply_cnt' in result.columns and 'shift_popularity' in result.columns:
        result['shift_popularity'] = (result['shift_popularity'] - result['hist_apply_cnt']).clip(lower=0)

    for pop in ['employer_popularity', 'location_popularity', 'task_popularity']:
        if pop in result.columns and 'hist_apply_cnt' in result.columns:
            result[pop] = (result[pop] - result['hist_apply_cnt']).clip(lower=0)

    # Заполняем NaN нулями
    for col in ['hist_view_cnt', 'hist_apply_cnt', 'hist_finished_cnt', 'hist_user_cancel_cnt', 'hist_system_cancel_cnt',
                'user_hist_views', 'user_hist_applies', 'user_hist_finished', 'user_hist_cancels',
                'shift_popularity', 'employer_popularity', 'location_popularity', 'task_popularity']:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    # - Признаки совместимости -
    result["location_match"] = (result["location_id"] == result["location_id_user"]).astype(int)
    result["mk_match"] = (result["need_mk"] == result["has_mk"]).astype(int)


    # - Временные признаки смены -
    dt = result["start_at"]
    result["shift_hour"] = dt.dt.hour
    result["shift_weekday"] = dt.dt.weekday
    result["shift_is_weekend"] = (result["shift_weekday"] >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["shift_hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["shift_hour"] / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * result["shift_weekday"] / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["shift_weekday"] / 7)

    # -Признаки оплаты и capacity -
    result["reward_per_hour"] = result["reward"] / (result["hours"] + 1e-5)
    result["reward_log"] = np.log1p(result["reward"])
    result["capacity_log"] = np.log1p(result["capacity"])
    result["capacity_ratio_to_10"] = result["capacity"] / 10.0
    result["needed_candidates"] = result["capacity"].clip(upper=10)

      # - as-of агрегаты по employer (успешные взаимодействия) -
    employer_success = merged_hist[merged_hist['interaction'].isin(['APPLY', 'FINISHED'])]\
        .groupby(['user_id', 'employer_id']).size().reset_index(name='user_employer_applies')
    result = result.merge(employer_success, on=['user_id', 'employer_id'], how='left')
    result['user_employer_applies'] = result['user_employer_applies'].fillna(0)

     # - as-of агрегаты по workplace -
    workplace_success = merged_hist[merged_hist['interaction'].isin(['APPLY', 'FINISHED'])]\
        .groupby(['user_id', 'workplace_id']).size().reset_index(name='user_workplace_applies')
    result = result.merge(workplace_success, on=['user_id', 'workplace_id'], how='left')
    result['user_workplace_applies'] = result['user_workplace_applies'].fillna(0)

     # - as-of агрегаты по task_type -
    task_success = merged_hist[merged_hist['interaction'].isin(['APPLY', 'FINISHED'])]\
        .groupby(['user_id', 'task_type']).size().reset_index(name='user_task_applies')
    result = result.merge(task_success, on=['user_id', 'task_type'], how='left')
    result['user_task_applies'] = result['user_task_applies'].fillna(0)

    # - recency-признаки (дни с последнего события) -
    last_view = merged_hist[merged_hist['interaction'] == 'VIEW']\
        .groupby('user_id')['ts'].max().reset_index(name='last_view_ts')
    last_apply = merged_hist[merged_hist['interaction'] == 'APPLY']\
        .groupby('user_id')['ts'].max().reset_index(name='last_apply_ts')
    last_finished = merged_hist[merged_hist['interaction'] == 'FINISHED']\
        .groupby('user_id')['ts'].max().reset_index(name='last_finished_ts')

    result = result.merge(last_view, on='user_id', how='left')
    result = result.merge(last_apply, on='user_id', how='left')
    result = result.merge(last_finished, on='user_id', how='left')
    result['days_since_last_view'] = (result['start_at'] - result['last_view_ts']).dt.days.fillna(9999)
    result['days_since_last_apply'] = (result['start_at'] - result['last_apply_ts']).dt.days.fillna(9999)
    result['days_since_last_finished'] = (result['start_at'] - result['last_finished_ts']).dt.days.fillna(9999)
    result['days_since_last_view'] = result['days_since_last_view'].clip(upper=999)
    result['days_since_last_apply'] = result['days_since_last_apply'].clip(upper=999)
    result['days_since_last_finished'] = result['days_since_last_finished'].clip(upper=999)

    # Пользователь считается cold-start, если у него нет ни одного APPLY в истории
    result['is_cold_start'] = (result['user_hist_applies'] == 0).astype(np.int8)
    result['is_warm_user'] = (1 - result['is_cold_start']).astype(np.int8)
    result['cold_x_loc_match'] = result['is_cold_start'] * result['location_match']
    result['cold_x_mk_compat'] = result['is_cold_start'] * result['mk_match']
    result['cold_x_strict_loc'] = result['is_cold_start'] * result['is_strict_location'].astype(int)
    # Совокупный score совместимости для cold-start
    result['cold_compatibility_score'] = (
        result['cold_x_loc_match'] * 2
        + result['cold_x_mk_compat'] * 1
        + result['cold_x_strict_loc'] * 1).astype(np.int8)

   
    # - Preference-признаки (исторические предпочтения пользователя) -
    merged_hist['shift_hour'] = merged_hist['start_at'].dt.hour
    merged_hist['shift_weekday'] = merged_hist['start_at'].dt.weekday
    pref_events = merged_hist[merged_hist['interaction'].isin(['APPLY', 'FINISHED'])].copy()
    if len(pref_events) > 0:
        # Средние значения по пользователю
        user_pref = pref_events.groupby('user_id', as_index=False).agg(
            user_avg_reward=('reward', 'mean'),
            user_avg_hours=('hours', 'mean'),
            user_avg_shift_hour=('shift_hour', 'mean'),
            user_avg_shift_weekday=('shift_weekday', 'mean'),
            user_pref_event_count=('interaction', 'count'))
        result = result.merge(user_pref, on='user_id', how='left')
        for col in ['user_avg_reward', 'user_avg_hours', 'user_avg_shift_hour', 'user_avg_shift_weekday', 'user_pref_event_count']:
            result[col] = result[col].fillna(0)
    else:
        for col in ['user_avg_reward', 'user_avg_hours', 'user_avg_shift_hour', 'user_avg_shift_weekday', 'user_pref_event_count']:
            result[col] = 0
    # Вычисляем меры близости к предпочтениям (только для пользователей с историей)
    result['reward_preference_match'] = np.where(result['user_pref_event_count'] > 0, 1.0 / (1.0 + np.abs(np.log1p(result['reward']) - np.log1p(result['user_avg_reward']))), 0.0).astype(np.float32)
    result['hours_preference_match'] = np.where(result['user_pref_event_count'] > 0, 1.0 / (1.0 + np.abs(result['hours'] - result['user_avg_hours'])), 0.0).astype(np.float32)
    # Циклическое расстояние для часа
    hour_diff = np.abs(result['shift_hour'] - result['user_avg_shift_hour'])
    hour_diff = np.minimum(hour_diff, 24 - hour_diff)
    result['hour_preference_match'] = np.where(result['user_pref_event_count'] > 0, np.maximum(0, 1.0 - hour_diff / 12.0), 0.0).astype(np.float32)
    # Циклическое расстояние для дня недели
    weekday_diff = np.abs(result['shift_weekday'] - result['user_avg_shift_weekday'])
    weekday_diff = np.minimum(weekday_diff, 7 - weekday_diff)
    result['weekday_preference_match'] = np.where(result['user_pref_event_count'] > 0, np.maximum(0, 1.0 - weekday_diff / 3.5), 0.0).astype(np.float32)
    

    # Удаляем лишние колонки
    result = result.rename(columns={'location_id': 'shift_location', 'location_id_user': 'user_location'})
    drop_cols = ["id", "employer_id", "workplace_id", "last_view_ts", "last_apply_ts", "last_finished_ts"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns], errors="ignore")

    return result
#строим целевую перменную
def _prepare_target_pairs(events: pd.DataFrame) -> pd.DataFrame:

    apply_events = events[events["interaction"] == "APPLY"][["user_id", "shift_id"]].drop_duplicates()
    cancel_events = events[events["interaction"] == "USER_CANCEL"][["user_id", "shift_id"]].drop_duplicates()
    apply_clean = apply_events.merge(cancel_events.assign(cancel=1), on=["user_id", "shift_id"], how="left")
    apply_clean = apply_clean[apply_clean["cancel"].isna()][["user_id", "shift_id"]]
    apply_clean["target"] = 1

    all_pairs = events[["user_id", "shift_id"]].drop_duplicates()
    target_df = all_pairs.merge(apply_clean, on=["user_id", "shift_id"], how="left")
    target_df["target"] = target_df["target"].fillna(0).astype(int)

    # Добавим время последнего события для каждой пары
    event_times = events.groupby(["user_id", "shift_id"])["ts"].max().reset_index()
    target_df = target_df.merge(event_times, on=["user_id", "shift_id"], how="left")
    return target_df
#разбиение по времени из baseline
def _time_split(frame: pd.DataFrame, test_ratio: float = 0.2, split_date: pd.Timestamp = None, align_to_week: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Training frame is empty after preprocessing.")
    unique_dates = np.sort(frame["start_at"].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough temporal points for split.")

    if split_date is None:
        split_idx = int(len(unique_dates) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(unique_dates) - 1))
        split_date = unique_dates[split_idx]
    else:
        split_date = pd.to_datetime(split_date)

    if align_to_week:
        # Округляем до понедельника (weekday 0 = Monday)
        days_to_subtract = split_date.weekday()  # 0=Monday, 6=Sunday
        split_date = split_date - pd.Timedelta(days=days_to_subtract)

    train = frame[frame["start_at"] <= split_date].copy()
    test = frame[frame["start_at"] > split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Time split produced empty train or test set.")

    #проверка на репрезантативность теста и трйена
    category_check = {}
    for col in ['task_type', 'employer_id']:
        if col in frame.columns:
            train_cats = set(train[col].dropna().unique())
            test_cats = set(test[col].dropna().unique())
            missing = train_cats - test_cats
            if missing:
                print(f"Предупреждение: в test отсутствуют категории {col}: {missing}")
            category_check[col] = {
                'train_categories': train_cats,
                'test_categories': test_cats,
                'missing_in_test': missing}

    return train, test
#функция обучения
def run_training(cfg: TrainConfig) -> dict:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Загрузка и валидация данных")
    users, shifts, events, checks = _load_and_validate_data(cfg)

    # - Разделение событий на train/test  -
    shifts_for_split = shifts[['id', 'start_at']].rename(columns={'id': 'shift_id'})
    train_shifts_df, test_shifts_df = _time_split(shifts_for_split, test_ratio=cfg.test_ratio, align_to_week=True)
    train_shifts = train_shifts_df['shift_id'].unique()
    test_shifts = test_shifts_df['shift_id'].unique()
    LOGGER.info(f"Train shifts: {len(train_shifts)}, Test shifts: {len(test_shifts)}")

    # - Фильтрация событий по сменам -
    train_events = events[events['shift_id'].isin(train_shifts)].copy()
    test_events = events[events['shift_id'].isin(test_shifts)].copy()
    LOGGER.info(f"Train events: {len(train_events)}, Test events: {len(test_events)}")

    # - Целевые пары -
    train_target = _prepare_target_pairs(train_events)
    test_target = _prepare_target_pairs(test_events)

    # - Построение признаков -
    LOGGER.info("Построение признаков для train (история = train)")
    train_frame = _build_features(users, shifts, train_events, train_target)
    LOGGER.info(f"Размер train_frame: {len(train_frame)}")

    LOGGER.info("Построение признаков для test (история = train, целевые пары = test)")
    test_frame = _build_features(users, shifts, train_events, test_target)
    LOGGER.info(f"Размер test_frame: {len(test_frame)}")

    train_frame['task_type'] = train_frame['task_type'].astype('category')
    test_frame['task_type'] = test_frame['task_type'].astype('category')


    # - Список признаков -
    numeric_features = [
        "user_hist_views", "user_hist_applies", "user_hist_finished", "user_hist_cancels",
        "shift_popularity", "employer_popularity", "location_popularity", "task_popularity",
        "shift_hour", "shift_weekday", "shift_is_weekend",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "reward", "hours", "capacity", "reward_per_hour", "reward_log",
        "capacity_log", "capacity_ratio_to_10", "needed_candidates",
        "location_match", "mk_match",
        "has_mk", "is_strict_location", "need_mk", "id_differential",
        "user_avg_reward", "user_avg_hours", "user_avg_shift_hour", "user_avg_shift_weekday",
        "user_pref_event_count", "reward_preference_match", "hours_preference_match",
        "hour_preference_match", "weekday_preference_match",
        "user_employer_applies", "user_workplace_applies", "user_task_applies",
        "days_since_last_view", "days_since_last_finished",
        "is_cold_start", "is_warm_user", "cold_x_loc_match", "cold_x_mk_compat", "cold_x_strict_loc",
        "cold_compatibility_score" ]


    categorical_features = ["task_type"]

    # Проверяем наличие колонок, при необходимости добавляем нули
    for col in numeric_features + categorical_features:
        if col not in train_frame.columns:
            LOGGER.warning(f"Column {col} missing in train_frame, adding dummy")
            train_frame[col] = 0
        if col not in test_frame.columns:
            test_frame[col] = 0

    X_train = train_frame[numeric_features + categorical_features].copy()
    y_train = train_frame["target"]
    X_test = test_frame[numeric_features + categorical_features].copy()
    y_test = test_frame["target"]

    # Преобразование булевых в int
    for col in ["has_mk", "is_strict_location", "need_mk", "id_differential"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0).astype(int)
            X_test[col] = X_test[col].fillna(0).astype(int)


    # - Обучение ансамбля -
    seeds = [42, 123, 456]
    models = []
    for seed in seeds:
      model = lgb.LGBMClassifier(
           n_estimators=150,
           max_depth=7,
           num_leaves=25,
           learning_rate=0.1,
           subsample=0.7,
           colsample_bytree=0.9,
          reg_lambda=1,
          reg_alpha=1,
          random_state=seed,
          n_jobs=1,
          verbose=-1)
      model.fit(X_train, y_train, categorical_feature=['task_type'])
      models.append(model)

    probas = [m.predict_proba(X_test)[:, 1] for m in models]
    proba_ensemble = np.mean(probas, axis=0)
    simple_auc = roc_auc_score(y_test, proba_ensemble)
    LOGGER.info(f"Simple ROC-AUC on test: {simple_auc:.4f}")

    metric_df = test_frame[["shift_id", "start_at", "capacity", "target"]].copy()
    metric_df["score"] = proba_ensemble
    metric_result = calculate_target_metric(metric_df)

    metrics = {
        "target_metric": metric_result.target_metric,
        "evaluated_days": metric_result.evaluated_days,
        "evaluated_groups": metric_result.evaluated_groups,
        "evaluated_shifts": metric_result.evaluated_shifts,
        "day_metrics": metric_result.day_metrics,
        "test_rows": len(test_frame),
        "train_rows": len(train_frame)}

    # - Сохранение артефактов -
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(models, f)



    train_proba_ensemble = np.mean([m.predict_proba(X_train)[:, 1] for m in models], axis=0)
    train_auc = roc_auc_score(y_train, train_proba_ensemble)
    LOGGER.info(f"Simple ROC-AUC on train: {train_auc:.4f}")
    LOGGER.info(f"Simple ROC-AUC on test:  {simple_auc:.4f}")
    print(f"Обычный ROC-AUC на обучении: {train_auc:.4f}")
    print(f"Обычный ROC-AUC на тесте:    {simple_auc:.4f}")

     # --- Сохранение артефактов для сервиса ---
    # shift_features (признаки смен)
    shift_feat_cols = [
        "shift_id", "start_at", "shift_location", "shift_hour", "shift_weekday",
        "shift_is_weekend", "capacity", "hours", "reward", "reward_per_hour",
        "need_mk", "id_differential", "shift_popularity", "employer_popularity",
        "location_popularity", "task_popularity", "capacity_log", "capacity_ratio_to_10",
        "needed_candidates", "has_mk", "is_strict_location"
    ]
    exist_shift_cols = [c for c in shift_feat_cols if c in train_frame.columns]
    shift_features_df = train_frame[exist_shift_cols].drop_duplicates("shift_id")
    joblib.dump(shift_features_df, output_dir / "shift_features.pkl")

    # user_stats (агрегированные характеристики пользователей)
    user_stats_cols = [
        "user_id", "user_location", "user_hist_views", "user_hist_applies",
        "user_hist_finished", "user_hist_cancels", "has_mk", "is_strict_location",
        "user_avg_reward", "user_avg_hours", "user_avg_shift_hour", "user_avg_shift_weekday",
        "user_pref_event_count", "reward_preference_match", "hours_preference_match",
        "hour_preference_match", "weekday_preference_match",
        "user_employer_applies", "user_workplace_applies", "user_task_applies",
        "days_since_last_view", "days_since_last_finished", "cold_compatibility_score",
        "is_cold_start", "is_warm_user", "cold_x_loc_match", "cold_x_mk_compat", "cold_x_strict_loc"
    ]
    exist_user_cols = [c for c in user_stats_cols if c in train_frame.columns]
    user_stats_df = train_frame[exist_user_cols].drop_duplicates("user_id")
    joblib.dump(user_stats_df, output_dir / "user_stats.pkl")

    # default_user_stats – нули для новых пользователей
    default_user = {col: 0 for col in user_stats_df.columns if col != "user_id"}
    joblib.dump(default_user, output_dir / "default_user_stats.pkl")

    # feature_columns – список признаков, используемых моделью
    joblib.dump(numeric_features + categorical_features, output_dir / "feature_columns.pkl")

    return {"metrics": metrics, "shap": {}}
