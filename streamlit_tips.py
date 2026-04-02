import streamlit as st
import pandas as pd
import joblib

# ======================
# 1. 모델 & 인코더 로드
# ======================
model = joblib.load("model/tips_model02.pkl")
encoders = joblib.load("model/tips_labelencoders02.pkl")

st.title("Tip Prediction App")

st.write("식사 정보 입력 후 팁 금액을 예측합니다.")

# ======================
# 2. 사용자 입력
# ======================
total_bill = st.number_input("총 금액 (total_bill)", 0.0, 100.0, 20.0)
size = st.slider("인원수 (size)", 1, 10, 2)

sex = st.selectbox("성별", encoders['sex'].classes_)
smoker = st.selectbox("흡연 여부", encoders['smoker'].classes_)
day = st.selectbox("요일", encoders['day'].classes_)
time = st.selectbox("식사 시간", encoders['time'].classes_)

# ======================
# 3. Feature Engineering (학습과 동일하게!)
# ======================
bill_per_person = total_bill / size
is_weekend = 1 if day in ['Sat', 'Sun'] else 0
is_dinner = 1 if time == 'Dinner' else 0
tip_rate = 0  # 예측 시에는 unknown → 0으로 처리

# ======================
# 4. 라벨 인코딩 적용
# ======================
sex_val = encoders['sex'].transform([sex])[0]
smoker_val = encoders['smoker'].transform([smoker])[0]
day_val = encoders['day'].transform([day])[0]
time_val = encoders['time'].transform([time])[0]

# ======================
# 5. 입력 데이터 구성
# ======================
input_data = pd.DataFrame({
    'total_bill': [total_bill],
    'sex': [sex_val],
    'smoker': [smoker_val],
    'day': [day_val],
    'time': [time_val],
    'size': [size],
    'bill_per_person': [bill_per_person],
    'is_weekend': [is_weekend],
    'is_dinner': [is_dinner],
    'tip_rate': [tip_rate]
})

# ======================
# 6. 예측
# ======================
if st.button("예측하기"):
    prediction = model.predict(input_data)[0]
    st.success(f"예상 팁 금액: ${prediction:.2f}")

    st.write("###입력 데이터")
    st.dataframe(input_data)