
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime

# 모델 로드 (캐시 적용)
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("kbs0035/my_fakenews_model")
    tokenizer = BertTokenizer.from_pretrained("kbs0035/my_fakenews_model")
    model.eval()
    return model, tokenizer

# 모델 불러오기
model, tokenizer = load_model()

# Streamlit UI 구성
st.title("허위정보 탐지 AI 서비스")

# 👉 새 접속 사용자에게 session_state 초기화 (자동)
if 'first_run' not in st.session_state:
    st.session_state['search_count'] = 0
    st.session_state['first_run'] = False

# 참여자 기본 정보 입력
st.subheader("1️⃣ 참여자 기본정보를 입력해 주세요")

user_id = st.text_input("참여코드 (본인 전화번호 끝 4자리 또는 임의 4자리)")
gender = st.radio("성별", ["남성", "여성", "기타/응답안함"])
age = st.number_input("나이 (숫자 입력)", min_value=10, max_value=100, step=1)
region = st.selectbox("거주지역", ["서울", "수도권(경기/인천)", "충청권", "영남권", "호남권", "강원/제주", "기타"])
political_ideology = st.slider("정치 이념 성향 (1 = 매우 진보적, 7 = 매우 보수적)", 1, 7, 4)
party_support = st.selectbox("현재 지지하는 정당", ["더불어민주당", "국민의힘", "정의당", "기타 정당", "지지 정당 없음"])

# 검색 횟수 카운트 (1~5개 제한)
st.write(f"현재 검색 횟수: {st.session_state['search_count']} / 5 (최소 1개 ~ 최대 5개까지 검색 가능)")

# 기사 입력
st.subheader("2️⃣ 기사 내용을 입력해 주세요")
user_input = st.text_area("기사 입력", height=150)

# 버튼 클릭 시 실행
if st.button("허위정보 탐색하기"):
    # 입력 확인
    if user_id.strip() == "" or user_input.strip() == "":
        st.warning("참여코드와 기사 내용을 모두 입력해 주세요.")
    elif st.session_state['search_count'] >= 5:
        st.warning("최대 5개까지 입력 가능합니다.")
    else:
        # 입력 텍스트 토크나이징
        inputs = tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

        # 모델 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][prediction].item() * 100

        # 예측 결과 표시
        if prediction == 1:
            st.error(f"❌ 허위 정보 가능성 높음. (신뢰도: {confidence:.2f}%)")
            result_text = "허위"
        else:
            st.success(f"✅ 진실된 정보 가능성 높음. (신뢰도: {confidence:.2f}%)")
            result_text = "진실"

        # 검색 로그 저장
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'gender': gender,
            'age': age,
            'region': region,
            'political_ideology': political_ideology,
            'party_support': party_support,
            'search_count': st.session_state['search_count'] + 1,
            'user_input': user_input,
            'result': result_text,
            'confidence': round(confidence, 2)
        }

        log_file = 'search_log.csv'

        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df_log = pd.DataFrame([log_entry])

        df_log.to_csv(log_file, index=False)
        st.info("검색 내용이 기록되었습니다.")
        
        # 검색 카운트 증가
        st.session_state['search_count'] += 1

# 검색 완료 안내
if st.session_state['search_count'] == 5:
    st.success("5개 입력 완료! 설문을 종료하셔도 됩니다.")

