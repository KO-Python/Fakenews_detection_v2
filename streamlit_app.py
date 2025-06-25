

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import dropbox

# ✅ 드롭박스 Access Token
DROPBOX_ACCESS_TOKEN = '여기에 넣으세요'

# ✅ 드롭박스 클라이언트 연결
dbx = dropbox.Dropbox('sl.u.AF1PourIfLqJOoaxZDsQLnm3HjeItiAVxU5O-DzpsHEwtovgVYU9keVdfxXmdjArTGlThqo7xGfAGx6OJ8sCBYoTBcjhnakyUKh4wna67g7_NC83rVtxUOIEKhF67mVKNFJatBQoN0_CFb2BtmAWsCIqG23Vc3qxCmVJrzsnZvLlR44UXFKBv9Eck2_0PWrObI_d_hXfF_eqY6N9mPu-rI9XCHFs5FZWR47vtHKsyzkxsk9cOD__aoib-BCTm9d1NHdnOvg_mn5HGwvcYqXAKQ4Jb79pAqIjYtWBdiLeIgiFE9qa-piLBCBDrNcguhnSrPhteUzJD8yy9jpFqSdwjeGOIFH4YWUFbgnJ5TSpcLt6YTC_HbF81Snkvh8wCCtLPvdjx1A8CgMIHw1ajfNw-pJmQx1M_MlZJSQM_vJGSW0NXEP8ytdbdG3I7rg0rjisTI1wkirKdgpkbaM7wKi9cw_4jvgob4fa7br3f5EPqBEBqOUvKlyTAt7m7UmvRtr-ecVLJqyWgsyLigLja7bvZg7FQzpTNdJkJWTO0gFq_nAQnNBcBlZ3ev8CGr1ZEV8Ht9zMyJhWxDKx_FTqfDVEUQUk9nbFzBPHIUOSo4iB25cwsGzxKAEElu3nReViKYZ9tcjN-tGbBVszIEYmR9_6qFWWRg4DYq8Z1dsWaz7tdHBPvANXvrzj8hldJ5W6QJhe6sJxxuez3_K6dqX03RfpdhSF2wonP0Q-23ozS8QKoq4APbqAG2XGwjKQhuixbdBmOyaZUfljzc_KyJUVnxl0NWWmJ8J-f8UKk9c7LbWWNHG3S6FjqWKzJy5rsVYzId9lxNJnWaOUQLClqj4EkEUWhrf8fALtX-JFlGZHT-W4c8NMWp8-KL-sjUVEJL-eM_bABE3Vo4_jusZN3SyODtsSXoP9sgOtuu4OWRfGkp1IV_0qwWh1xyCXC_3U3Qgur3KCVk_MK7LsVqNm9i3Fz7cJFulydYzlypIxq5e85pNOmfQK05RHedwDD8MpT2KcKaPUCekwHBpnh5OfirK9lXOrQRfkyAlDP_yO_7NpcF1YcGyRMouH_GbMJtKBf3NK6CUc6-2jTxkVO4GyVz1nVv0UZDGxfk7xgdcXdsf2NKB4uL7P4aJWz7fWoob-i3tCfGR-WbfaA8DveMLcY-7b8lPvMkNhMsAJcHopAaF-4sEGzHsMvswpSoXf70OHefeHfsnmvTaM-FpvvRxxLnSrL3nfmiDhXYKuTnLCuIf208A7KDFLiWtocuR_3Hw-mqjU-A35oQ4')

# ✅ 모델 로드 (캐시 적용)
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("kbs0035/my_fakenews_model")
    tokenizer = BertTokenizer.from_pretrained("kbs0035/my_fakenews_model")
    model.eval()
    return model, tokenizer

# ✅ 모델 불러오기
model, tokenizer = load_model()

# ✅ Streamlit UI 구성
st.title("허위정보 탐지 AI 서비스")

# ✅ 초기화 버튼
if st.button("초기화"):
    st.session_state.clear()
    st.experimental_rerun()

# ✅ 참여자 기본 정보 입력
st.subheader("1️⃣ 참여자 기본정보를 입력해 주세요")

user_id = st.text_input("참여코드 (본인 전화번호 끝 4자리 또는 임의 4자리)")
gender = st.radio("성별", ["남성", "여성", "기타/응답안함"])
age = st.number_input("나이 (숫자 입력)", min_value=10, max_value=100, step=1)
region = st.selectbox(
    "거주지역", 
    ["선택하세요", "서울", "수도권(경기/인천)", "충청권", "영남권", "호남권", "강원/제주", "기타"]
)
political_ideology = st.slider("정치 이념 성향 (1 = 매우 진보적, 10 = 매우 보수적)", 1, 10, 5)
party_support = st.selectbox("현재 지지하는 정당", ["더불어민주당", "국민의힘", "정의당", "기타 정당", "지지 정당 없음"])

# ✅ 검색 횟수 카운트 (1~5개 제한)
if 'search_count' not in st.session_state:
    st.session_state['search_count'] = 0

st.write(f"현재 검색 횟수: {st.session_state['search_count']} / 5 (최소 1개 ~ 최대 5개까지 검색 가능)")

# ✅ 기사 입력
st.subheader("2️⃣ 기사 내용을 입력해 주세요")
user_input = st.text_area("기사 입력", height=150)

# ✅ 버튼 클릭 시 실행
if st.button("허위정보 탐색하기"):
    # 입력 확인
    if user_id.strip() == "" or user_input.strip() == "" or region == "선택하세요":
        st.warning("참여코드, 거주지역, 기사 내용을 모두 입력해 주세요.")
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

        # ✅ 검색 로그 저장
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

        # ✅ 드롭박스에 업로드
        try:
            with open("search_log.csv", "rb") as f:
                dbx.files_upload(f.read(), "/FakeNews/search_log.csv", mode=dropbox.files.WriteMode.overwrite)
            st.success("✅ 드롭박스 저장 완료!")
        except Exception as e:
            st.error(f"❌ 드롭박스 저장 실패: {e}")

        # ✅ 검색 카운트 증가
        st.session_state['search_count'] += 1

# ✅ 검색 완료 안내
if st.session_state['search_count'] == 5:
    st.success("5개 입력 완료! 설문을 종료하셔도 됩니다.")


