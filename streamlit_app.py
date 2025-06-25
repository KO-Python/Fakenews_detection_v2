

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import gspread_dataframe as gd

# 구글 시트 ID (고정)
SHEET_ID = "1oZy6Nkvcice2Xs9mrA6fS0vy0pp5Ay3iSN0Z73KeCVk"

# 인증
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_dict(
    json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"]),
    scope
)

gc = gspread.authorize(creds)

# 구글 시트 접근
sheet = gc.open_by_key(SHEET_ID).worksheet("Sheet1")

# 모델 로드
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("kbs0035/my_fakenews_model")
    tokenizer = BertTokenizer.from_pretrained("kbs0035/my_fakenews_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# UI
st.title("허위정보 탐지 AI 서비스")

# 검색 카운트
if 'search_count' not in st.session_state:
    st.session_state['search_count'] = 0

# 참여자 기본 정보
st.subheader("1️⃣ 참여자 기본정보를 입력해 주세요")

user_id = st.text_input("참여코드 (본인 전화번호 끝 4자리 또는 임의 4자리)")
gender = st.radio("성별", ["남성", "여성", "기타/응답안함"])
age = st.number_input("나이 (숫자 입력)", min_value=10, max_value=100, step=1)
region = st.selectbox("거주지역", ["서울", "수도권(경기/인천)", "충청권", "영남권", "호남권", "강원/제주", "기타"])
political_ideology = st.slider("정치 이념 성향 (1 = 매우 진보적, 10 = 매우 보수적)", 1, 10, 5)
party_support = st.selectbox("현재 지지하는 정당", ["더불어민주당", "국민의힘", "정의당", "기타 정당", "지지 정당 없음"])

st.write(f"현재 검색 횟수: {st.session_state['search_count']} / 5 (최소 1개 ~ 최대 5개까지 가능)")

# 기사 입력
st.subheader("2️⃣ 기사 내용을 입력해 주세요")
user_input = st.text_area("기사 입력", height=150)

# 허위정보 탐색 버튼
if st.button("허위정보 탐색하기"):
    if user_id.strip() == "" or user_input.strip() == "":
        st.warning("참여코드와 기사 내용을 모두 입력해 주세요.")
    elif st.session_state['search_count'] >= 5:
        st.warning("최대 5개까지 입력 가능합니다.")
    else:
        # 모델 예측
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

        # 로그 기록
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

        # 구글 시트에 기록
        df_existing = pd.DataFrame(sheet.get_all_records())
        df_new = pd.concat([df_existing, pd.DataFrame([log_entry])], ignore_index=True)
        gd.set_with_dataframe(sheet, df_new)
        st.info("검색 내용이 Google Sheet에 기록되었습니다.")

        # 카운트 증가
        st.session_state['search_count'] += 1

# 종료 버튼
if st.session_state['search_count'] >= 1:
    if st.button("설문 종료하기"):
        st.success("참여해주셔서 감사합니다.")

