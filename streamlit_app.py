

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import dropbox

# ✅ 드롭박스 Access Token (선생님이 새로 발급 후 여기에 넣으시면 됩니다)
DROPBOX_ACCESS_TOKEN = 'sl.u.AF2IfaMHeFvyeZwxy8QB5prGjCUMF2HX6ghef98nNZMBdhZZn0ZePztY2iguCN6Qvg4IX2ib6hlSJcI5T9UjwcYVe3ATvlJmaN451pBTLgvsrJWX785UhBjey3M13w2qiVdpGAB2fgl8bKrmikP1vgQSvf73toCzLnwy_T8Awr7xAGSk952oqCuIREk5KW_2_4vVmK3eREkVfEwy8y0_zxuo1g88dTG1ibc2SSFJszMv4iQCh-Q-AE7oYrc5xeHIEg4NGbDLK5k0kMr82oAVFJ3hmty8ApBHyn7ozRAHYny7-4g1oebcyTrpHe1P3iGwOV-MRsAFdtOJKQ3_gcaePNOnWWMCGMf8En4HInGHpxWY43K4rRbHyPxkTrAX9r6Pva8ItsKkeXWsLytCNlOxqcdSQxEpZpVw9R8_31cNxZX5wQVUGFfWq-sfKne5Sf3TB55arNNMJZFEuOza9Dxmm8q0bQD-PXUIzKrOo8764w9UxNc3zZhE1G8UIFDqiWwUrwf7i-dlysBEj__T3AVprH5q21ZOddVRnTE1S519D21slhQq00e6flLWOsMDtaWBY1pRMFRDKA97f1hSqoL_NYPTCfdT9oykNM7NHdI5Ym5atW1iGGU31QGrZDZKk0LaIeIZF46NXW5H5PG9XlUrr5ViwCgimguL4wAgV3Z_XA1kUg3EcRA1XffiWvSlLIDQX_SbspcfOXI1_cUtHxeXebiZ1aRPbvaw0x9YyECCHf_7cSazQIrGJQR9MrUqkaNvNdV13AU3-uxAhXe5RSVeE70zql4-qtLh0SNafa9-aXpbt3FfykyOlkOjpL6yveYg1Kfy9Xddv0wHL1fhL3t34U1pDJBM7DdVUDJvXzNmOaYz1L5OVZFgidCQ84a1w4xrDbJB-KPUG43MY_wDT0JQUF1VpAJ-0kHg6n4CP-s5aDaswZATvj7yK_OPiktYSOP6on2r3s6PWsqlwyd84wpCEnfxnWlwVI0ZdMcd1o290oQMIz710bwANgqtBcpxwQFAcjL1dF19q_QcKUjLIWoVj2unnNCHzyrDij4cXvwxMliiThZavB1v53ZNyAyJpxRnW1sTy-r5_xYJq0OCw4fbP5aWDP2nWVZTmQ2gL-jYZXM6EKfGuuoWGApH2eHYQ_rnDOxUlaLFqa75BHFLIze1MF-WgIQ6UjG2BhvSRfwPwr8ZDPNX-GaDeQUfY6DOkV3MnG-VYWARxHDqXeE5iTxNsPKLiWZxE0XVFsfP4s6XVdgV72kohMJ9KAZlGJovERNW3fc'

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



