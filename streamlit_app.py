
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import json
from datetime import datetime

# ✅ 모델 로드
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("kbs0035/my_fakenews_model")
    tokenizer = BertTokenizer.from_pretrained("kbs0035/my_fakenews_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ✅ Google Sheets 인증 (Secrets 사용)
scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]

creds = ServiceAccountCredentials.from_json_keyfile_dict(
    json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"]),
    scope
)
gc = gspread.authorize(creds)
sheet = gc.open("FakenewsDetection_Logs").sheet1

# ✅ UI 구성
st.title("허위정보 탐지 AI 서비스")

# 검색 카운트
if 'search_count' not in st.session_state:
    st.session_state['search_count'] = 0

# 초기화 버튼
if st.button("초기화"):
    st.session_state.clear()
    st.rerun()

# ✅ 참여자 정보 입력
st.subheader("1️⃣ 참여자 기본정보를 입력해 주세요")

user_id = st.text_input("참여코드 (본인 전화번호 끝 4자리 또는 임의 4자리)")
gender = st.radio("성별", ["남성", "여성", "기타/응답안함"], index=-1)
age = st.number_input("나이 (숫자 입력)", min_value=10, max_value=100, step=1)
region = st.selectbox("거주지역", ["", "서울", "수도권(경기/인천)", "충청권", "영남권", "호남권", "강원/제주", "기타"])
political_ideology = st.slider("정치 이념 성향 (1 = 매우 진보적, 10 = 매우 보수적, 5 = 중도)", 1, 10, 5)
party_support = st.selectbox("현재 지지하는 정당", ["", "더불어민주당", "국민의힘", "정의당", "기타 정당", "지지 정당 없음"])

st.write(f"현재 검색 횟수: {st.session_state['search_count']} / 5 (최소 1개 ~ 최대 5개까지 검색 가능)")

# ✅ 기사 입력
st.subheader("2️⃣ 기사 내용을 입력해 주세요")
user_input = st.text_area("기사 입력", height=150)

# ✅ 허위정보 탐색하기
if st.button("허위정보 탐색하기"):
    if user_id.strip() == "" or user_input.strip() == "":
        st.warning("참여코드와 기사 내용을 모두 입력해 주세요.")
    elif st.session_state['search_count'] >= 5:
        st.warning("최대 5개까지 입력 가능합니다.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][prediction].item() * 100

        if prediction == 1:
            st.error(f"❌ 허위 정보 가능성 높음. (신뢰도: {confidence:.2f}%)")
            result_text = "허위"
        else:
            st.success(f"✅ 진실된 정보 가능성 높음. (신뢰도: {confidence:.2f}%)")
            result_text = "진실"

        # ✅ 로그 저장
        log_entry = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id,
            gender,
            age,
            region,
            political_ideology,
            party_support,
            st.session_state['search_count'] + 1,
            user_input,
            result_text,
            round(confidence, 2)
        ]

        sheet.append_row(log_entry)

        st.info("검색 내용이 기록되었습니다.")
        st.session_state['search_count'] += 1

# ✅ 종료 버튼
if st.session_state['search_count'] >= 1:
    if st.button("설문 종료하기"):
        st.success("설문을 종료하셨습니다. 감사합니다!")

