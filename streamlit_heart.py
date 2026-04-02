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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "heart_disease_uci_korean.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "heart_model.pkl"
ENC_PATH = MODEL_DIR / "heart_encoders.pkl"

DROP_COLS = ["주요 혈관 수", "심장 혈류 상태"]
COLS_TO_FIX = ["안정 시 혈압", "나이", "최대 심박수"]

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
    """노트북과 동일: 결측·파생·타겟·라벨 인코딩까지 (df는 이미 DROP_COLS 제거된 상태)."""
    df = df.copy()
    for col in COLS_TO_FIX:
        df[col] = df[col].replace(0, np.nan)

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

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
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["target"] = (df["심장병 진단 결과"] > 0).astype(int)

    encoders: dict[str, LabelEncoder] = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


@functools.lru_cache(maxsize=1)
def load_or_train_model() -> tuple[RandomForestClassifier, dict[str, LabelEncoder]]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and ENC_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(ENC_PATH)

    df = pd.read_csv(DATA_PATH, encoding="cp949")
    df = df.drop(columns=DROP_COLS)
    df, encoders = _run_notebook_pipeline(df)

    X = df.drop(columns=["심장병 진단 결과", "target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENC_PATH)
    return model, encoders


@functools.lru_cache(maxsize=1)
def load_training_frame_for_stats() -> pd.DataFrame:
    """평균·최빈값은 학습 데이터와 동일한 방식으로 계산."""
    df = pd.read_csv(DATA_PATH, encoding="cp949")
    df = df.drop(columns=DROP_COLS)
    for col in COLS_TO_FIX:
        df[col] = df[col].replace(0, np.nan)
    return df


def _fill_stats_from_training() -> tuple[pd.Series, dict[str, object]]:
    df = load_training_frame_for_stats()
    num_means = df.select_dtypes(include=["float64", "int64"]).mean()
    obj_modes = {c: df[c].mode(dropna=True).iloc[0] for c in df.select_dtypes(include=["object"]).columns}
    return num_means, obj_modes


def build_single_row_from_inputs(user: dict) -> pd.DataFrame:
    """사용자 입력 1행 → 노트북과 동일 전처리·인코딩 (심장병 진단 결과 없음)."""
    model, encoders = load_or_train_model()
    num_means, obj_modes = _fill_stats_from_training()

    df = pd.DataFrame([user])
    for col in COLS_TO_FIX:
        df[col] = df[col].replace(0, np.nan)

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(num_means[col])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(obj_modes[col])

    for col in ("공복 혈당 여부", "운동 유발성 협심증"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["나이 대비 혈압 비율"] = df["안정 시 혈압"] / df["나이"]
    df["고령자 여부"] = (df["나이"] >= 60).astype(int)
    df["최대 심박수 대비 현재 혈압 비율"] = df["최대 심박수"] / df["안정 시 혈압"]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = encoders[col].transform(df[col].astype(str))

    return df


def main() -> None:
    st.set_page_config(page_title="심장질환 위험 예측", layout="wide")
    st.title("심장질환 위험 예측")
    st.caption("노트북 `김민수_머신러닝과제.ipynb`와 동일한 전처리·RandomForest 설정을 사용합니다.")

    if not DATA_PATH.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    model, encoders = load_or_train_model()

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
        proba = model.predict_proba(X_one)[0]
        pred = int(model.predict(X_one)[0])

        st.subheader("예측 결과")
        label = "질환 위험(양성) 가능성이 상대적으로 높음" if pred == 1 else "정상(음성)에 가까운 패턴"
        st.success(f"**예측 클래스:** {'질환 의심 (1)' if pred == 1 else '정상 (0)'} — {label}")
        st.write(f"**질환(1) 확률:** {proba[1]:.1%}  |  **정상(0) 확률:** {proba[0]:.1%}")

        with st.expander("전처리 후 특성 벡터 (14차원)"):
            st.dataframe(X_one, use_container_width=True)

    st.divider()
    st.info(
        "교육·과제용 참고 모델이며 실제 의료 진단을 대체하지 않습니다. "
        "모델 파일이 없으면 첫 실행 시 데이터로 학습 후 `model/`에 저장합니다."
    )


if __name__ == "__main__":
    main()
