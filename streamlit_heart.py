"""
심장질환 위험 예측 (김민수_머신러닝과제.ipynb 파이프라인과 동일)
실행: streamlit run streamlit_heart.py
"""
from __future__ import annotations

import functools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "heart_disease_uci_korean.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "heart_model.pkl"
ENC_PATH = MODEL_DIR / "heart_encoders.pkl"
SCALER_PATH = MODEL_DIR / "heart_scaler.pkl"

DROP_COLS = ["주요 혈관 수", "심장 혈류 상태"]
# notebook에서 0을 '미기록/결측'으로 보고 NaN 처리한 컬럼들
COLS_TO_FIX = ["안정 시 혈압", "콜레스테롤", "최대 심박수"]

# 사용자 입력 대상 (타겟·삭제 컬럼 제외)
INPUT_FEATURE_COLS = [
    "나이",
    "성별",
    "표준진료지침",
    "안정 시 혈압",
    "콜레스테롤",
    "공복 혈당 여부",
    "안정 시 심전도 결과",
    "최대 심박수",
    "운동 유발성 협심증",
    "운동 후 심전도 저하 수치(0)",
    "운동 후 심전도 회복 패턴(상향)",
]


def _run_notebook_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    for col in COLS_TO_FIX:
        df[col] = df[col].replace(0, np.nan)

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # fillna 후 TRUE/FALSE 등이 bool로 내려가면 object가 아니게 되어 인코더에서 빠짐 → 문자열로 고정
    for col in ("공복 혈당 여부", "운동 유발성 협심증"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["나이 대비 혈압 비율"] = df["안정 시 혈압"] / df["나이"]
    df["고령자 여부"] = (df["나이"] >= 60).astype(int)
    df["최대 심박수 대비 현재 혈압 비율"] = df["최대 심박수"] / df["안정 시 혈압"]
    # 파생변수(비율)에서 inf가 발생할 수 있으므로 해당 컬럼만 median으로 보정
    for _col in ["나이 대비 혈압 비율", "최대 심박수 대비 현재 혈압 비율"]:
        df[_col] = df[_col].replace([np.inf, -np.inf], np.nan)
        df[_col] = df[_col].fillna(df[_col].median())
        df[_col] = df[_col].fillna(0)
    df["target"] = (df["심장병 진단 결과"] > 0).astype(int)

    encoders: dict[str, LabelEncoder] = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


@functools.lru_cache(maxsize=1)
def load_or_train_model():
   
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and ENC_PATH.exists():
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENC_PATH)
        scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
        return model, encoders, scaler

    df = pd.read_csv(DATA_PATH, encoding="cp949")
    df = df.drop(columns=DROP_COLS)
    df, encoders = _run_notebook_pipeline(df)

    X = df.drop(columns=["심장병 진단 결과", "target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENC_PATH)
    return model, encoders, None


@functools.lru_cache(maxsize=1)
def load_training_frame_for_stats() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="cp949")
    df = df.drop(columns=DROP_COLS)
    for col in COLS_TO_FIX:
        df[col] = df[col].replace(0, np.nan)
    return df


def _fill_stats_from_training() -> tuple[pd.Series, dict[str, object]]:
    df = load_training_frame_for_stats()
    num_medians = df.select_dtypes(include=["float64", "int64"]).median()
    obj_modes = {c: df[c].mode(dropna=True).iloc[0] for c in df.select_dtypes(include=["object"]).columns}
    return num_medians, obj_modes


def build_single_row_from_inputs(user: dict) -> pd.DataFrame:
    _, encoders, _ = load_or_train_model()
    num_medians, obj_modes = _fill_stats_from_training()

    df = pd.DataFrame([user])
    for col in COLS_TO_FIX:
        df[col] = df[col].replace(0, np.nan)

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(num_medians[col])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(obj_modes[col])

    for col in ("공복 혈당 여부", "운동 유발성 협심증"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["나이 대비 혈압 비율"] = df["안정 시 혈압"] / df["나이"]
    df["고령자 여부"] = (df["나이"] >= 60).astype(int)
    df["최대 심박수 대비 현재 혈압 비율"] = df["최대 심박수"] / df["안정 시 혈압"]
    # 파생변수(비율)에서 inf가 발생할 수 있으므로 해당 컬럼만 median으로 보정
    for _col in ["나이 대비 혈압 비율", "최대 심박수 대비 현재 혈압 비율"]:
        df[_col] = df[_col].replace([np.inf, -np.inf], np.nan)
        df[_col] = df[_col].fillna(df[_col].median())
        df[_col] = df[_col].fillna(0)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = encoders[col].transform(df[col].astype(str))

    return df


def main() -> None:
    st.set_page_config(page_title="심장질환 위험 예측", layout="wide")
    st.title("심장질환 위험 예측")

    if not DATA_PATH.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    model, encoders, scaler = load_or_train_model()

    st.sidebar.title("메뉴")
    page = st.sidebar.radio("페이지 선택", ["예측하기", "상세 데이터 분석"])

    if page == "상세 데이터 분석":
        # 1) 검증 정확도(Accuracy) 계산
        num_medians, obj_modes = _fill_stats_from_training()

        df_raw = pd.read_csv(DATA_PATH, encoding="cp949")
        df_raw = df_raw.drop(columns=DROP_COLS)

        for col in COLS_TO_FIX:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].replace(0, np.nan)

        # 결측치 채우기(노트북과 동일한 방식: 수치 median / 범주 최빈값)
        numeric_cols = df_raw.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if col in num_medians.index:
                df_raw[col] = df_raw[col].fillna(num_medians[col])

        object_cols = df_raw.select_dtypes(include=["object"]).columns
        for col in object_cols:
            if col in obj_modes:
                df_raw[col] = df_raw[col].fillna(obj_modes[col])

        # 파생변수 생성
        df_raw["나이 대비 혈압 비율"] = df_raw["안정 시 혈압"] / df_raw["나이"]
        df_raw["고령자 여부"] = (df_raw["나이"] >= 60).astype(int)
        df_raw["최대 심박수 대비 현재 혈압 비율"] = df_raw["최대 심박수"] / df_raw["안정 시 혈압"]

        # 비율 파생변수 inf/NaN 정리
        for _col in ["나이 대비 혈압 비율", "최대 심박수 대비 현재 혈압 비율"]:
            df_raw[_col] = df_raw[_col].replace([np.inf, -np.inf], np.nan)
            df_raw[_col] = df_raw[_col].fillna(df_raw[_col].median())
            df_raw[_col] = df_raw[_col].fillna(0)

        # 타겟 생성
        df_raw["target"] = (df_raw["심장병 진단 결과"] > 0).astype(int)

        # 범주형 인코딩(저장된 encoders를 그대로 사용)
        for col in df_raw.select_dtypes(include=["object"]).columns:
            if col in encoders:
                df_raw[col] = encoders[col].transform(df_raw[col].astype(str))

        X = df_raw.drop(columns=["심장병 진단 결과", "target"])
        y = df_raw["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 스케일러 적용(로지스틱 회귀용)
        if scaler is not None:
            if hasattr(scaler, "feature_names_in_"):
                X_test = X_test.reindex(columns=list(scaler.feature_names_in_), fill_value=0)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("모델 검증 결과")
        st.metric("Accuracy (테스트 세트)", f"{acc:.4f}")

        st.divider()

        # 2) 분류에 중요한 변수 Top 2
        st.subheader("분류에 중요한 변수 Top 2")

        if hasattr(model, "coef_"):
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                feature_names = list(scaler.feature_names_in_)
            else:
                feature_names = list(X.columns)

            coef_abs = pd.Series(np.abs(model.coef_[0]), index=feature_names)
            top2 = coef_abs.sort_values(ascending=False).head(2)
        elif hasattr(model, "feature_importances_"):
            feature_names = list(X.columns)
            imp = pd.Series(model.feature_importances_, index=feature_names)
            top2 = imp.sort_values(ascending=False).head(2)
        else:
            top2 = pd.Series(dtype=float)

        if len(top2) == 0:
            st.warning("중요 변수 계산이 불가능한 모델 형태입니다.")
        else:
            top2_df = top2.reset_index()
            top2_df.columns = ["변수", "중요도(계수 절댓값/중요도)"]
            st.dataframe(top2_df, use_container_width=True)
            st.bar_chart(top2)

            st.caption("로지스틱 회귀의 경우 계수 절댓값 기준으로 Top 2를 산출했습니다.")

        st.info("상세 데이터 분석은 교육·과제용으로, 입력 값에 대한 개인별 설명이 아닌 전체 데이터 기반 요약입니다.")
        return

    st.subheader("환자 정보 입력")
    cols = st.columns(2)
    user: dict = {}

    with cols[0]:
        user["나이"] = st.number_input("나이", min_value=1, max_value=120, value=55, step=1)
        user["성별"] = st.selectbox("성별", list(encoders["성별"].classes_))
        user["표준진료지침"] = st.selectbox("표준진료지침(흉통 유형)", list(encoders["표준진료지침"].classes_))
        user["안정 시 혈압"] = st.number_input("안정 시 혈압", min_value=0.0, max_value=300.0, value=130.0, step=1.0)
        user["콜레스테롤"] = st.number_input("콜레스테롤", min_value=0.0, max_value=600.0, value=250.0, step=1.0)
        user["공복 혈당 여부"] = st.selectbox("공복 혈당 여부", list(encoders["공복 혈당 여부"].classes_))

    with cols[1]:
        user["안정 시 심전도 결과"] = st.selectbox(
            "안정 시 심전도 결과", list(encoders["안정 시 심전도 결과"].classes_)
        )
        user["최대 심박수"] = st.number_input("최대 심박수", min_value=0.0, max_value=250.0, value=150.0, step=1.0)
        user["운동 유발성 협심증"] = st.selectbox(
            "운동 유발성 협심증", list(encoders["운동 유발성 협심증"].classes_)
        )
        user["운동 후 심전도 저하 수치(0)"] = st.number_input(
            "운동 후 심전도 저하 수치(0)", min_value=0.0, max_value=20.0, value=1.0, step=0.1
        )
        user["운동 후 심전도 회복 패턴(상향)"] = st.selectbox(
            "운동 후 심전도 회복 패턴(상향)",
            list(encoders["운동 후 심전도 회복 패턴(상향)"].classes_),
        )

    missing = [c for c in INPUT_FEATURE_COLS if c not in user]
    if missing:
        st.warning(f"입력 누락: {missing}")
        return

    if st.button("예측하기", type="primary"):
        X_one = build_single_row_from_inputs(user)
        # notebook이 LogisticRegression(GridSearch + StandardScaler)로 학습했다면 scaler를 적용해야 함
        # 또한 StandardScaler가 fit된 컬럼 순서와 동일해야 하므로 reindex로 맞춤.
        if scaler is not None:
            if hasattr(scaler, "feature_names_in_"):
                X_one = X_one.reindex(columns=list(scaler.feature_names_in_), fill_value=0)
            X_in = scaler.transform(X_one)
        else:
            X_in = X_one
        proba = model.predict_proba(X_in)[0]
        pred = int(model.predict(X_in)[0])

        st.subheader("예측 결과")
        label = "질환 위험 가능성이 상대적으로 높음" if pred == 1 else "정상에 가까운 패턴"
        st.success(f"**예측 클래스:** {'질환 의심' if pred == 1 else '정상 '} — {label}")
        st.write(f"**질환 확률:** {proba[1]:.1%}  |  **정상 확률:** {proba[0]:.1%}")

        with st.expander("입력된 정보를 확인하세요"):
            st.dataframe(X_one, use_container_width=True)

    st.divider()

if __name__ == "__main__":
    main()
