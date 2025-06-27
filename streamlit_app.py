import streamlit as st
import torch
from transformers import BertTokenizer
from model import MultiOutputBERT

# 모델 로드
@st.cache_resource(show_spinner=False)
def load_model():
    model = MultiOutputBERT(
        pretrained_model_name='klue/bert-base',
        num_category_labels=8,
        num_target_labels=12
    )
    model.load_state_dict(torch.load("saved_model/hate_speech_model.pt", map_location='cpu'))
    model.eval()
    return model

# 토크나이저 및 모델 불러오기
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = load_model()

# 라벨 리스트
category_list = ['공격', '모욕', '배제', '비방', '비하', '조롱', '증오', '폄하']
target_list = ['공무원', '기독교', '남성', '성소수자', '아시아인', '여성',
               '이민자', '이슬람교', '장애인', '정치인', '청소년', '흑인']

# UI 구성
st.title("🛡️ 혐오 표현 탐지기")

text = st.text_area(
    "문장을 입력하세요:",
    placeholder="예: 정치인은 존재 자체가 불쾌하다."
)

st.caption("📝 예: 온라인 커뮤니티나 소셜미디어에서 접한 혐오 표현이 의심되는 문장을 입력해보세요.")

if st.button("분석하기"):
    if not text.strip():
        st.warning("⚠️ 분석할 문장을 입력해주세요.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        with torch.no_grad():
            cat_logits, tar_logits = model(**inputs)
            cat_pred = torch.argmax(cat_logits, dim=1).item()
            tar_pred = torch.argmax(tar_logits, dim=1).item()

        st.success(f"📂 Category: **{category_list[cat_pred]}**")
        st.success(f"🎯 Target: **{target_list[tar_pred]}**")