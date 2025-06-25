

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import dropbox

# ✅ 드롭박스 Access Token (선생님이 새로 발급 후 여기에 넣으시면 됩니다)
DROPBOX_ACCESS_TOKEN = 'sl.u.AF14tlmf5wnusvJnLa7kmT8mhNd5G4ekeFYkve5LYOrR5wHuX256d4MG41tED1a7kTvg097bKBYtWErJrJSK321o83r9raa_ich1aunfPRlS1KWy8k1fbqTTa6AJWmxi3Q2mrcInJMCRuT2TP1ksvQUMzOy2GX_pRhiYnv6FLc_XYl_eUM8jAp-bmfbPj5DsqypKPAcqU8z7zEvg84KWtkyq5wCUbMSdPsgh7qQGEdPxLTMrZby4yaAiS-T6LLPiu4MZAVfhNhoPOqJxzlp1cJOFuPkwiGfOP16M9YzDwYCEvnaxIvBagGDepJJAKvAsvcVQfmtuNKFaxa-XIEr6Cw3QnOIUbQeh7pwlRX_8Qa4VDhJErXz9tQlM_ZIlxRBcqtUXYfwTDSFHg4HgwdyY0qejUwT5T31s-Q3iEDZOZUAU5KymLA6O21kcQHeUNOnFyaWtGhwhOWMT5LZOe3ITJn7WYIZg1VESl4XedEZa_Vl_K2kQilSBSl-YstxuYTez9uiQsZVBMHr6d9-h4ojrSl2cewPG4Ty579C0Yli3ry56wu0nyt8p6FTtyFZP1j7brtS_D70hT3FPseMcuIwuVub8jQaqPdIAVMbzvv1DstvdXV6542hdYrXPSWJmo1GmSrVk6YWmpQDkXqz--LQfAQ8ruAN1_uZCAx2p6ZgP4I3HccCzqs47WqrHjnCYBJ4fFZR4RB-JRDdshkCp1fwS8ZM3Sb-7o3fmla2uyjHn5JUOnU5PqrjaIn0mc1w8QpjS0rspFIXhzCOmHkpD7L4Sn3rQFrDC5bAXWCOOhro3aJd2uKAdCk6j3rDV1eEu6KdMdWxrPBNvzTM5CFcpeBEntsLuIo29XYkO7clX6P4fckitqvw6HZ4Lu_DgSskVOxKAwm2L4Htf_5281k2lrpOTCqj_QyXyDGbPXt36Qy1NM6fWj6YwRekr4fhUcx2x2ZV5q7OBPAukjxbJx3ROV9-BlzOlAH8z9PWpM3AkRRrcKjzIOmyZlpmPahNI73SIpM1NciG96tccWz3GO14bLDLcMmMFmq06DVt5CNnp8fJIkoiqnC_jS8VIu1JI6GIjIfnUh3o8hjcDPT1GaGjQjyTBN_91AI9ANEIAcuGc_3pm-HbZvk87A0yU9lYHS35-E8jtCW1q-6y_ufo4FnNgVKvkuR4aW3rUtqbXnl0MLKbXo9l2NUzYVBGo51GGKCRGnjBI2En1H6nbiQoo2AgKK34xd7MEKh7j-XrfZJMWgz7y-ds7Wsl7D70A68Igow1Yg6mhvhQ'

# ✅ 드롭박스 클라이언트 연결
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

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

# ✅ 참여자 기본 정보 입력
st.subheader("참여자 기본정보를 입력해 주세요")

user_id = st.text_input("참여코드 (본인 전화번호 끝 4자리 또는 임의 4자리)")

gender = st.selectbox("성별", ["선택하세요", "남성", "여성", "기타/응답안함"])

age = st.number_input("나이 (숫자 입력)", min_value=10, max_value=100, step=1)

region = st.selectbox(
    "거주지역",
    ["선택하세요", "서울", "수도권(경기/인천)", "충청권", "영남권", "호남권", "강원/제주", "기타"]
)

political_ideology = st.slider(
    "정치 이념 성향 (1 = 매우 진보적, 10 = 매우 보수적) → 이동해서 선택해주세요", 1, 10, 5
)

party_support = st.selectbox(
    "현재 지지하는 정당",
    ["선택하세요", "더불어민주당", "국민의힘", "정의당", "기타 정당", "지지 정당 없음"]
)

# ✅ 검색 횟수 카운트 (1~5개 제한)
if 'search_count' not in st.session_state:
    st.session_state['search_count'] = 0

st.write(f"현재 검색 횟수: {st.session_state['search_count']} / 5 (최소 1개 ~ 최대 5개까지 검색 가능)")

# ✅ 기사 입력
st.subheader("내용을 입력해 주세요")
user_input = st.text_area("기사 입력", height=150)

# ✅ 버튼 클릭 시 실행
if st.button("허위정보 탐색하기"):
    # 입력 확인
    if user_id.strip() == "" or gender == "선택하세요" or region == "선택하세요" or party_support == "선택하세요" or user_input.strip() == "":
        st.warning("⚠️ 참여코드, 성별, 거주지역, 지지정당, 기사 내용을 모두 입력해 주세요.")
    elif st.session_state['search_count'] >= 5:
        st.warning("⚠️ 최대 5개까지 입력 가능합니다.")
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
        st.info("✅ 검색 내용이 기록되었습니다.")

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
    st.success("🎉 5개 입력 완료! 설문을 종료하셔도 됩니다.")



