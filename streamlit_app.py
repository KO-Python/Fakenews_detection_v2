
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import dropbox
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ✅ 드롭박스 Access Token
DROPBOX_ACCESS_TOKEN = 'sl.u.AF2O1qcOi_HfmrVsSslCVLsd_52ldGchuW5EcjiOKk5dXADJeFhHwsEDl_w6FN3UksRTwdPKp2Z3l_a7oocNyBD2mvoGKbcOIUj_A0gzSnqfeTYU3f9sH3yi88mV9474Cllz2pPFIpGr7jUZzuoL5jBKFH-WfeGOiQCt18tL8XnHpNR2diUq2NwW6uYbi8H3L1k3s1hScKaqsR_95hqkwsX0UYUtZAwwj17PEW2LZz2bDQ0GbS6fyf9azRbL4--UNilnC_AZ2Z9ZK2FXOaXZHkK92tqC8VQy9z9_5Xw8TChaTNBgAqBVzN4u7MK82EfmBONUv09Y-cNtdiYsGl_1BcuK4wYcR5W-OK_rY4hysrxrYU9s9OwsScOAz16bZ_gj1LV0EYKLXXAM1LbW1O9hK2RXE1fhRoov_gEMQoiZCjXtnXcnWd8DdXNMoW_R1DbWPvm0o9tfGjfgTxaC9v_uzT_YhXPw8ScKVe3pn4-43YPFOzjCR3mk4Ip6Q_rhgSC27ANaOt9Nk5vWmNEompMPc8iWp1rBLRWkTOfVpmXZsFCQUCuSh6dd42amoQz4vk7BUclldJNu8u-VfiUkrMpijUSg5gJND5xLaGZO4d-6BfzHCmdo7Dv-gPm11kDfYmwAKuvKerf3wyrPikF606htFFyNKAkIQEqwCyO6yE3Ny2W9PatySlYjb681ewh6t_Pab8MZWYLclVy7Hj3AkGbHeeI4Q-MWBJOhjdoGvKmmCyM5pHePyNLgLW68MCpg18ANV9JcxlBGZctQx6I6lypqjuAxAnKjK_9tfn67Yx6MKnb5A1fQX9LwWqhWJFJlis3VSRvYOFajeCmsHutf6caCkOFr4KtSBTQgo9PKERYmLXoBXznAKtMQ6vuogrrd_zw-m-ndnerhI6iy3wQbBP2S8gzMN0wX_JVFy0RJZmreHJK8J0-ejy7LbbcDEdHWpZA0unsihl0BYJ3D_3CATJiGpAaccL_QFsKylvB_dJYUfCWZiIbo9Sq_c1RnKj8MpyUZioAa41MJdITVsINCdMmFqcod4Tyz2eVNsckTExE3-GCLnyEKKqF46nQdN8E85ESkEu4dxGS_v-9dQPiSTBqVoo7vrUkSmyeyxbmLMBajueU1RarqCKgePxuHObKQ2ntyNpUUEn8a2B3YVtTyGD_C3bSOYNO1sz-xWu_G8eUulEtkZ2g0tj5EM0sFQGJBVrZZeClt3caZPdteEV9eXOuYO_YBhFngkq9uOx23KMpc5kAlZo0UwDu8T8N5uZ9XGCoA4tI'

# ✅ 드롭박스 클라이언트 연결
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# ✅ 모델 로드 (캐시 적용)
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("kbs0035/hate_classify_model")
    tokenizer = BertTokenizer.from_pretrained("kbs0035/hate_classify_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ✅ Streamlit UI 구성
st.title("혐오 표현 판별 AI 서비스")

st.info('''참여 안내
- 본 서비스는 텍스트 내 혐오 표현 여부 및 그 유형과 대상을 자동으로 판별합니다.
- 온라인에서 보거나 들은 혐오적 또는 차별적인 문장을 입력해 주세요.
- 예시: "저런 XX 같은 여자들이 문제야"
''')

user_id = st.text_input("참여코드 (전화번호 끝 4자리 또는 임의 4자리)")

if 'entry_count' not in st.session_state:
    st.session_state['entry_count'] = 0

st.write(f"현재 입력 횟수: {st.session_state['entry_count']} / 3")

sentence = st.text_area("문장을 입력해 주세요", height=150)

if st.button("판별하기"):
    if user_id.strip() == "" or sentence.strip() == "":
        st.warning("참여코드와 문장을 입력해 주세요.")
    elif st.session_state['entry_count'] >= 3:
        st.warning("최대 3개까지 입력 가능합니다.")
    else:
        inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][prediction].item() * 100

        # 예시 분류
        categories = ['조롱', '비방', '비하', '폄하', '공격', '증오', '중립']
        targets = ['여성', '남성', '성소수자', '이주민', '장애인', '노인', '청년', '종교 집단', '정치 성향', '지역', '국적/인종', '기타']

        predicted_category = categories[prediction % len(categories)]
        predicted_target = targets[prediction % len(targets)]

        st.write(f"분류 결과: **{predicted_category}**, 대상: **{predicted_target}** (신뢰도: {confidence:.2f}%)")

        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'sentence': sentence,
            'category': predicted_category,
            'target': predicted_target,
            'confidence': round(confidence, 2)
        }

        log_file = 'hate_detected.csv'
        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df_log = pd.DataFrame([log_entry])

        df_log.to_csv(log_file, index=False)
        st.success("결과가 저장되었습니다.")

        try:
            with open(log_file, "rb") as f:
                dbx.files_upload(f.read(), f"/HateClassifier/{log_file}", mode=dropbox.files.WriteMode.overwrite)
            st.success("드롭박스 저장 완료!")
        except Exception as e:
            st.error(f"드롭박스 저장 실패: {e}")

        st.session_state['entry_count'] += 1

if st.session_state['entry_count'] == 3:
    st.success("3개 입력 완료! 감사합니다.")


