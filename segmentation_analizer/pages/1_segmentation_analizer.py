import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_image_zoom import image_zoom

from torchvision import transforms
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import onnxruntime

st.set_page_config(page_title="Segmentation Infer Analysis", layout="wide")
st.markdown("# Segmentation Infer Analysis")

# === 세션 상태 초기화 ===
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

imge_size=(224,224)
uploaded_files = st.file_uploader("",type=["png", "jpg", "jpeg"], accept_multiple_files=True,key=f"uploader_{st.session_state.upload_key}")

# === 초기화 버튼 ===
if st.button("🗑️ 업로드된 이미지 초기화"):
    st.session_state.upload_key += 1  # 먼저 키를 증가
    st.session_state.uploaded_files = []
    st.rerun()

# 업로드한 파일 저장
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# === model ===
model = onnxruntime.InferenceSession("./deeplabv3_resnet101.onnx", providers=["CPUExecutionProvider"])

# --- (추가) 모델 출력으로 클래스 수 자동 추론 시도 ---
def _autodetect_class_num(ort_session):
    try:
        out_shape = ort_session.get_outputs()[0].shape  # 대개 [1, C, H, W] 또는 [N, C, ...]
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            return int(out_shape[1])
    except Exception:
        pass
    return None

_auto_cn = _autodetect_class_num(model)

# === 사이드바 옵션들 ===
st.sidebar.header("옵션")
threshold = st.sidebar.slider("1. 컨피던스 임계값", 0.0, 1.0, 0.5, step=0.01)
alpha = st.sidebar.slider("2. 마스크 불투명도", 0.0, 1.0, 0.3, step=0.05)
view_mode = st.sidebar.radio("3. 보기 모드", ["전체 마스크", "클래스별 마스크"])

# --- (변경) 클래스 개수 지정: 자동 추론 값이 있으면 기본값으로 사용 ---
class_num = st.sidebar.number_input(
    "4. 클래스 개수", min_value=1, max_value=512,
    value=int(_auto_cn) if _auto_cn else 21, step=1
)

selected_class = st.sidebar.selectbox("5. 클래스 선택", list(range(class_num)))

# --- (변경) 색상표: class_num에 맞춰 자동 생성 ---
def build_color_map(n_classes: int):
    base = [
        (0,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
        (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0),
        (128,0,128),(0,128,128),(192,192,192),(128,128,128),
        (255,165,0),(0,100,0),(75,0,130),(255,105,180),
        (139,69,19),(173,216,230),(46,139,87)
    ]
    if n_classes <= len(base):
        return {i: base[i] for i in range(n_classes)}
    # HSV로 추가 색 생성
    cmap = {i: base[i] for i in range(len(base))}
    extra = n_classes - len(base)
    for k in range(extra):
        hue = int(180 * (k / max(1, extra)))  # OpenCV HSV hue: 0~179
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
        cmap[len(base)+k] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return cmap

color_map = build_color_map(class_num)

def infer(model,uploaded_file):
    input_image = Image.open(uploaded_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(list(imge_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # (참고) onnxruntime은 torch.cuda / model.to('cuda')와 무관합니다. 아래 코드는 불필요.
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    with torch.no_grad():
        ort_inputs = {model.get_inputs()[0].name: input_batch.numpy()}
        ort_outs = model.run(None, ort_inputs)
        output = ort_outs[0]
        output = output.squeeze(0)  # (C,H,W)

    input_image = np.array(input_image).astype(np.uint8)
    resized_image = cv2.resize(input_image, imge_size, interpolation=cv2.INTER_LINEAR)
    return output,resized_image

# === 추론 ===
if st.session_state.uploaded_files:
    filenames = [file.name for file in uploaded_files]
    selected_file = st.selectbox("🔍 추론할 이미지를 선택하세요", filenames)

    for uploaded_file in uploaded_files:
        if uploaded_file.name == selected_file:

            output,input_image = infer(model,uploaded_file)

            if view_mode == "클래스별 마스크":
                prob_map = output[selected_class]
                mask_bin = (prob_map > threshold).astype(np.uint8)
                mask_rgb = np.zeros_like(input_image)
                color = color_map[selected_class]
                for i in range(3):
                    mask_rgb[:, :, i] = mask_bin * color[i]
                # 전체 비율 표는 클래스별 보기에서는 생략
                percentages = None
            else:
                pred_mask = output.argmax(0)
                mask_rgb = np.zeros_like(input_image)
                total_pixels = pred_mask.size
                percentages = {}
                for class_id in range(class_num):
                    class_pixels = np.count_nonzero(pred_mask == class_id)
                    percentages[class_id] = (class_pixels / total_pixels) * 100
                    color = color_map[class_id]
                    for i in range(3):
                        mask_rgb[:, :, i][pred_mask == class_id] = color[i]

            # 오버레이 이미지 생성
            blended = cv2.addWeighted(input_image, 1 - alpha, mask_rgb, alpha, 0)

            # 컬럼 분리
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### 🖼️ {uploaded_file.name}")
                image_zoom(blended)
            with col2:
                if percentages is not None:
                    df_percent = pd.DataFrame({
                        "Class ID": list(percentages.keys()),
                        "Pixel %": [round(v,2) for v in percentages.values()]
                    })
                    st.markdown("### 📊 클래스별 픽셀 비율")
                    st.dataframe(df_percent)

            # 마스크 저장
            mask_save = Image.fromarray(mask_rgb)
            buf = BytesIO()
            mask_save.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="📥 마스크 저장 (PNG)",
                data=buf,
                file_name=f"{selected_file[:-4]}_mask.png",
                mime="image/png"
            )
            break
