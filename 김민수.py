import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 1. 데이터 로드
@st.cache_data
def load_data():
    data = pd.read_csv('./dataset/HR_comma_sep.csv')
    data.rename(columns={'Departments ': 'Department'}, inplace=True)
    # 범주형 변수 변환
    data = pd.get_dummies(data, columns=['Department', 'salary'], drop_first=True)
    return data

data = load_data()


# 2. 특성 선택
X = data[['satisfaction_level', 'number_project', 'time_spend_company']]
y = data['left']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  3. 데이터 스케일링 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# streamlit 화면 구성
st.title('퇴사 여부 예측 시스템 구현')
st.markdown('직원의 정보를 입력하면 퇴사 가능성을 에측합니다.')

# 입력 위젯 배치 및 값 저장
a = st.slider('만족도', min_value=0.0, max_value=1.0, step=0.1)
b = st.number_input('프로젝트 수', min_value=1, max_value=10, step=1)
c = st.number_input('회사 경력', min_value=1, max_value=20, step=1)

acc = accuracy_score(y_test, model.predict(X_test_scaled))

st.sidebar.metric("모델 정확도", f"{acc:.2f}")

# 예측 실행 버튼
if st.button('예측 실행') :
    # 1. 입력 데이터를 모델이 학습한 순서대로 리스트로 만들기
    input_data = [[a,b,c]]

    # 2. 데이터 스케일링
    input_scaled = scaler.transform(input_data)

    # 3. 예측 수행
    prediction = model.predict(input_scaled)

    # 퇴사/잔류 확률 가져오기
    proba = model.predict_proba(input_scaled)

    st.subheader('예측 결과')
    
    # 4. 결과 표시
    if prediction[0] == 1 :
        # 퇴사 예측 
        st.error(f'퇴사 가능성이 높습니다. 퇴사 가능성 : {proba[0][1]:.1%}')
    else :
        # 잔류 예측
        st.success(f'퇴사 가능성이 낮습니다. 잔류 가능성 : {proba[0][0]:.1%}')

# 시각화
if st.checkbox('퇴사 가능성 시각화') :

    # 중요도 데이터 생성
    importances = model.feature_importances_
    feature_names = X.columns

    # Streamlit 중요도 시각화
    st.bar_chart(dict(zip(feature_names, importances)))

# 히트맵
if st.checkbox('히트맵') :
    st.write("==== 히트맵 ====")
    
    subset_cols = ['satisfaction_level', 'number_project', 'time_spend_company', 'left']
    subset_data = data[subset_cols]

    corr_matrix = subset_data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', linewidths=5)
    ax.set_title('변수 간 상관관계 히트맵')
    st.pyplot(fig)



# 혼동 행렬
if st.checkbox('혼동 행렬') :
    st.write("==== 혼동 행렬 ====")
    
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['잔류', '퇴사'])

    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    ax.set_title('혼동 행렬')
    st.pyplot(fig)