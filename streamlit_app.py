

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import dropbox

# ✅ 드롭박스 Access Token
DROPBOX_ACCESS_TOKEN = 'sl.u.AF2O1qcOi_HfmrVsSslCVLsd_52ldGchuW5EcjiOKk5dXADJeFhHwsEDl_w6FN3UksRTwdPKp2Z3l_a7oocNyBD2mvoGKbcOIUj_A0gzSnqfeTYU3f9sH3yi88mV9474Cllz2pPFIpGr7jUZzuoL5jBKFH-WfeGOiQCt18tL8XnHpNR2diUq2NwW6uYbi8H3L1k3s1hScKaqsR_95hqkwsX0UYUtZAwwj17PEW2LZz2bDQ0GbS6fyf9azRbL4--UNilnC_AZ2Z9ZK2FXOaXZHkK92tqC8VQy9z9_5Xw8TChaTNBgAqBVzN4u7MK82EfmBONUv09Y-cNtdiYsGl_1BcuK4wYcR5W-OK_rY4hysrxrYU9s9OwsScOAz16bZ_gj1LV0EYKLXXAM1LbW1O9hK2RXE1fhRoov_gEMQoiZCjXtnXcnWd8DdXNMoW_R1DbWPvm0o9tfGjfgTxaC9v_uzT_YhXPw8ScKVe3pn4-43YPFOzjCR3mk4Ip6Q_rhgSC27ANaOt9Nk5vWmNEompMPc8iWp1rBLRWkTOfVpmXZsFCQUCuSh6dd42amoQz4vk7BUclldJNu8u-VfiUkrMpijUSg5gJND5xLaGZO4d-6BfzHCmdo7Dv-gPm11kDfYmwAKuvKerf3wyrPikF606htFFyNKAkIQEqwCyO6yE3Ny2W9PatySlYjb681ewh6t_Pab8MZWYLclVy7Hj3AkGbHeeI4Q-MWBJOhjdoGvKmmCyM5pHePyNLgLW68MCpg18ANV9JcxlBGZctQx6I6lypqjuAxAnKjK_9tfn67Yx6MKnb5A1fQX9LwWqhWJFJlis3VSRvYOFajeCmsHutf6caCkOFr4KtSBTQgo9PKERYmLXoBXznAKtMQ6vuogrrd_zw-m-ndnerhI6iy3wQbBP2S8gzMN0wX_JVFy0RJZmreHJK8J0-ejy7LbbcDEdHWpZA0unsihl0BYJ3D_3CATJiGpAaccL_QFsKylvB_dJYUfCWZiIbo9Sq_c1RnKj8MpyUZioAa41MJdITVsINCdMmFqcod4Tyz2eVNsckTExE3-GCLnyEKKqF46nQdN8E85ESkEu4dxGS_v-9dQPiSTBqVoo7vrUkSmyeyxbmLMBajueU1RarqCKgePxuHObKQ2ntyNpUUEn8a2B3YVtTyGD_C3bSOYNO1sz-xWu_G8eUulEtkZ2g0tj5EM0sFQGJBVrZZeClt3caZPdteEV9eXOuYO_YBhFngkq9uOx23KMpc5kAlZo0UwDu8T8N5uZ9XGCoA4tI'

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
st.info('''참여 안내
- 본 서비스는 연구 목적으로 제공됩니다.
- 최근 접한 정보 또는 뉴스 중에서 특정 정치인 또는 정당에 대해 '진실'이라고 믿는 내용을 1문장으로 작성해 주세요.
- 예시: "윤석열 대통령은 청년 일자리 확대를 위해 연 100만 개의 일자리를 창출했다."
- 최대 3개까지 입력 가능하며, 1개만 입력하고 종료해도 됩니다.
''')


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

