

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import dropbox

# ✅ 드롭박스 Access Token
DROPBOX_ACCESS_TOKEN = 'sl.u.AF3XnPnBw8o3nWyPHw-Wluvb9gzy5VFeXWgn20xDaFp33HeMYJn63W5DRYOx71c3JSAhxVLbDdFCD8p6n7DPgSXVO8L4F9cqoQqV1ks18sRjEihcI1PnelcMqaCGULClWofoTJ-utzXHCe-ydGq5uH9O_FSeTiVdq07-ivBH70FTCazpLEp3xxKzFe2WqUXpao8oNU5gO-v4oMVezmC2YeAiv81c6aAoPQEMs2WVnq3n6wDGT95Gr7qCUdBvGgUjtk7ifPbafJ_X_ywNu7jK-9i6A2zMHuVp-8WKvpPWlV5j_IG3rAnqj_sky4a6F7D_yAv4cS0y_TMVKo_v1GXZd_7f3oBD2OfBxls0Kne4Pe3JeVRNJP_f59g8_VBXOPnpxJ-ESwwqU_SI_gVuWfUWk6bqLnP_oSNKTd1C__q4xb4wY9GNGkWTU9ci3NXpWli6cixEKxj-6J3JrvCXiJohrkmDsOMWdWN5yFaQj656YOHddE-OePwZi_IVV3tIg7qQgTh4TOfo68k3rKgptXY5MUcWzRyntDXb-Ry7ZewjNCJh6xvNxnYFilL3QpQpTkCrZ9DuXkofs5umnRR9wwv9ZM1kqdGAowyO8joGbtFLXsmCs43U79zBfBlA2bQ45sEb7RqEx7LWttwTKfefdyYjmpQQ__t11JOXQc_N7EbNh7OukEcL0TzO19GTlOBVFrJN2XOUGhqY41i0JSa1chHpfkb6zJQiCPI2qPKb19VNvuSH6VjQX57ZQRfYxi375fXkz-bbdX4FSmZoMjc9S0wIIsENlS5icStdKPV0papHdfapbKQS4fnkjWHAr70OL5z_H4eBsL7yDgsQmu_dfneUkyV_zcTyO8bLo09oGZwVdGfD1v3clR4uNv8bh8W2e98F0bplra6ySfx6CVSaaVjTfgQJD6R9a_y4ZVFMarD3uYpQctYgmrTd_4rtCo6auNnLXopdsq-ULJR8N8Fda1QaBooDrGnhwjyZHsmz9EPfsoR3ZHysiX7SxehYXIT8YQ5klCtP5gHe9M_4eupDjCoUGiqtv2yqpkpksPMuv28FZ75GuD7X0rIagCE_CA4flVML4L-eiT3W78zMfx3xtClrTbE6Ws8IS7YVQE2xRKIg-GR_jqSsPkRRP03bHyg_5x87kUAf8xtblAe28bgEgCUtPkCT3H3leW5iJqjVH5uDFkzwKw195BURkKFcu7bMlVSSprs0yPT15NUOyzt-dCtiy7uiaOdTnVFQYdFkcdli9QYyXGfXcQm2eefJZjUGD-R_EO8'

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

# ✅ 안내 메시지
st.info(
    "참여 안내
"
    "- 본 서비스는 연구 목적으로 제공됩니다.
"
    "- 최근 접한 정보 또는 뉴스 중에서 특정 정치인 또는 정당에 대해 '진실'이라고 믿는 내용을 1문장으로 작성해 주세요.
"
    "- 예시: "윤석열 대통령은 청년 일자리 확대를 위해 연 100만 개의 일자리를 창출했다."
"
    "- 최대 3개까지 입력 가능하며, 1개만 입력하고 종료해도 됩니다.")

# ✅ 참여코드 입력
user_id = st.text_input("참여코드 (전화번호 끝 4자리 또는 임의 4자리)")

# ✅ 검색 횟수 카운트 (1~3개 제한)
if 'search_count' not in st.session_state:
    st.session_state['search_count'] = 0

st.write(f"현재 입력 횟수: {st.session_state['search_count']} / 3")

# ✅ 기사 입력
user_input = st.text_area("내용 입력", height=150)

# ✅ 버튼 클릭 시 실행
if st.button("탐색하기"):
    if user_id.strip() == "" or user_input.strip() == "":
        st.warning("참여코드와 내용을 모두 입력해 주세요.")
    elif st.session_state['search_count'] >= 3:
        st.warning("최대 3개까지 입력 가능합니다.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][prediction].item() * 100

        if prediction == 1:
            st.error(f"허위 정보 가능성 높음. (신뢰도: {confidence:.2f}%)")
            result_text = "허위"
        else:
            st.success(f"진실된 정보 가능성 높음. (신뢰도: {confidence:.2f}%)")
            result_text = "진실"

        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
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
        st.info("입력 내용이 저장되었습니다.")

        try:
            with open("search_log.csv", "rb") as f:
                dbx.files_upload(f.read(), "/FakeNews/search_log.csv", mode=dropbox.files.WriteMode.overwrite)
            st.success("드롭박스 저장 완료!")
        except Exception as e:
            st.error(f"드롭박스 저장 실패: {e}")

        st.session_state['search_count'] += 1

if st.session_state['search_count'] == 3:
    st.success("3개 입력 완료! 감사합니다.")

