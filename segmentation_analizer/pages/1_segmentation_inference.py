import os
import numpy as np
import cv2
import torch
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO

from torchvision import transforms
from streamlit_image_zoom import image_zoom

# =========================================================
# i18n: language state & translations
# =========================================================
if "lang" not in st.session_state:
    st.session_state.lang = "ko"  # 기본 한국어

TR = {
    "ko": {
        "page_title": "Segmentation Infer Analysis",
        "title": "Segmentation Infer Analysis",
        "language": "언어",
        "lang_ko": "한국어",
        "lang_en": "English",

        "options": "옵션",
        "threshold": "1. 컨피던스 임계값",
        "alpha": "2. 마스크 불투명도",
        "view_mode": "3. 보기 모드",
        "view_all": "전체 마스크",
        "view_class": "클래스별 마스크",
        "class_count": "4. 클래스 개수",
        "class_select": "5. 클래스 선택",

        "reset_uploaded_img":"🗑️ 업로드된 이미지 초기화",

        "pick_image": "🔍 추론할 이미지를 선택하세요",
        "image_title": "### 🖼️ {name}",
        "table_title": "### 📊 분석 결과",
        "col_class_id": "클래스 ID",
        "col_pixel_pct": "픽셀(%)",
        "col_avg_conf": "평균 Confidence",

        "download_mask": "📥 마스크 저장 (PNG)",
        "onnx_loaded": "ONNX 모델 로드: {path}",
        "onnx_fallback": "ONNX 로드 실패 → PyTorch에서 DeepLabV3-ResNet101 모델 다운로드. 이유: {err}",
        "torch_loaded": "PyTorch DeepLabV3-ResNet101 모델 다운로드 완료",
        "torch_fail": "PyTorch 모델 로드도 실패했습니다: {err}",
        "warn_set_model": "먼저 모델을 설정하세요.",
        "sel_class_metric": "선택 클래스 평균 confidence",

        "tab_all": "전체 이미지 분석",
        "tab_single": "단일 이미지 분석",
        "summary_title": "### 🧮 요약 결과 (전체 이미지 집계)",
        "per_image_title": "### 🔎 이미지별 상세",
        "download_summary": "📥 요약 CSV 다운로드",
        "download_detail": "📥 상세 CSV 다운로드",

        "summary_title_short": "### 요약 결과",
        "detail_title_short": "### 이미지별 상세 테이블",

        "col_image": "이미지",
        "col_classes": "클래스",
        "col_avg_conf_over_present": "평균 신뢰도(%)",
        "cls_col_prefix": "클래스 ",
        "cls_col_suffix": " 픽셀(%)",
        "avg_conf_suffix": " 평균 신뢰도(%)",

    },
    "en": {
        "page_title": "Segmentation Infer Analysis",
        "title": "Segmentation Infer Analysis",
        "language": "Language",
        "lang_ko": "Korean",
        "lang_en": "English",

        "options": "Options",
        "threshold": "1. Confidence Threshold",
        "alpha": "2. Mask Opacity",
        "view_mode": "3. View Mode",
        "view_all": "All Masks",
        "view_class": "Class Mask",
        "class_count": "4. Classes",
        "class_select": "5. Select Class",

        "reset_uploaded_img":"🗑️ Clear uploaded images",
        "pick_image": "🔍 Select an image to infer",
        "image_title": "### 🖼️ {name}",
        "table_title": "### 📊 Analysis Results",
        "col_class_id": "Class ID",
        "col_pixel_pct": "Pixel(%)",
        "col_avg_conf": "Avg Confidence",

        "download_mask": "📥 Download Mask (PNG)",
        "onnx_loaded": "Loaded ONNX model: {path}",
        "onnx_fallback": "Failed to load ONNX → download from PyTorch. Reason: {err}",
        "torch_loaded": "The PyTorch DeepLabV3-ResNet101 model was downloaded successfully.",
        "torch_fail": "Failed to load PyTorch model: {err}",
        "warn_set_model": "Please set the model first.",
        "sel_class_metric": "Avg confidence of selected class",

        "tab_all": "All Images Analysis",
        "tab_single": "Single Image Analysis",
        "summary_title": "### 🧮 Summary (Aggregated over all images)",
        "per_image_title": "### 🔎 Per-image Breakdown",
        "download_summary": "📥 Download Summary CSV",
        "download_detail": "📥 Download Detail CSV",

        "summary_title_short": "### Summary",
        "detail_title_short": "### Per-image Details",
        "col_image": "Image",
        "col_classes": "Classes",
        "col_avg_conf_over_present": "Avg confidence(%)",
        "cls_col_prefix": "class",
        "cls_col_suffix": " pixel(%)",
        "avg_conf_suffix": " avg_confidence(%)",

    },
}
def t(key: str) -> str:
    return TR[st.session_state.lang][key]

st.set_page_config(page_title=t("page_title"), layout="wide")
st.markdown(f"# {t('title')}")

# --- Language switcher (sidebar) ---
lang_choice = st.sidebar.radio(
    t("language"),
    (TR["ko"]["lang_ko"], TR["en"]["lang_en"]),
    horizontal=True,
    index=0 if st.session_state.lang == "ko" else 1,
)
st.session_state.lang = "ko" if lang_choice == TR["ko"]["lang_ko"] else "en"

# =========================================================
# Session state init
# =========================================================
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# =========================================================
# Uploader
# =========================================================
imge_size = (224, 224)
uploaded_files = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.upload_key}",
)

if st.button(t('reset_uploaded_img')):
    st.session_state.upload_key += 1
    st.session_state.uploaded_files = []
    st.rerun()

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# =========================================================
# Model loading: ONNX → fallback to PyTorch
# =========================================================
MODEL_BACKEND = None  # "onnx" or "torch"
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

ONNX_PATH = "./pages/deeplabv3_resnet101.onnx"

# Try ONNX first
try:
    import onnxruntime
    if not os.path.isfile(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")
    model = onnxruntime.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    MODEL_BACKEND = "onnx"
    st.info(t("onnx_loaded").format(path=ONNX_PATH))
except Exception as e:
    st.warning(t("onnx_fallback").format(err=e))
    # Fallback: PyTorch torchvision
    try:
        import torchvision
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        ).to(device).eval()
        MODEL_BACKEND = "torch"
        st.success(t("torch_loaded"))
    except Exception as e2:
        st.error(t("torch_fail").format(err=e2))
        st.stop()

# =========================================================
# Autodetect class num
# =========================================================
def _autodetect_class_num_onnx(ort_session):
    try:
        out_shape = ort_session.get_outputs()[0].shape  # [N,C,H,W]
        if len(out_shape) >= 2 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            return int(out_shape[1])
    except Exception:
        pass
    return None

def _autodetect_class_num_torch(pt_model):
    try:
        # torchvision deeplabv3 classifier last Conv
        return int(pt_model.classifier[-1].out_channels)
    except Exception:
        return None

_auto_cn = _autodetect_class_num_onnx(model) if MODEL_BACKEND == "onnx" else _autodetect_class_num_torch(model)

# =========================================================
# Sidebar options (i18n labels)
# =========================================================
st.sidebar.header(t("options"))
threshold = st.sidebar.slider(t("threshold"), 0.0, 1.0, 0.5, step=0.01)
alpha = st.sidebar.slider(t("alpha"), 0.0, 1.0, 0.7, step=0.05)

# 내부 값 'all'/'class'로 관리 (표시는 다국어)
view_mode = st.sidebar.selectbox(
    t("view_mode"),
    options=["all", "class"],
    format_func=lambda x: t("view_all") if x == "all" else t("view_class"),
)

class_num = st.sidebar.number_input(
    t("class_count"),
    min_value=1, max_value=512,
    value=int(_auto_cn) if _auto_cn else 21, step=1
)
selected_class = st.sidebar.selectbox(t("class_select"), list(range(class_num)))

# =========================================================
# Color map
# =========================================================
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
    cmap = {i: base[i] for i in range(len(base))}
    extra = n_classes - len(base)
    for k in range(extra):
        hue = int(180 * (k / max(1, extra)))  # 0~179 (OpenCV HSV)
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        cmap[len(base)+k] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return cmap
color_map = build_color_map(class_num)

# =========================================================
# Softmax over channel (C,H,W)
# =========================================================
def softmax_channelwise(logits_chw: np.ndarray) -> np.ndarray:
    shifted = logits_chw - logits_chw.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)

# =========================================================
# Inference
# =========================================================
def infer(model_obj, uploaded_file, backend: str):
    input_image = Image.open(uploaded_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(list(imge_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)  # (3,H,W)

    if backend == "onnx":
        # ONNX Runtime path
        input_batch = input_tensor.unsqueeze(0).numpy()  # (1,3,H,W)
        input_name = model_obj.get_inputs()[0].name
        ort_outs = model_obj.run(None, {input_name: input_batch})
        output = ort_outs[0].squeeze(0)  # (C,H,W) numpy
    else:
        # PyTorch path
        input_batch = input_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
        with torch.no_grad():
            out = model_obj(input_batch)['out'][0]  # (C,H,W) torch.Tensor
        output = out.detach().cpu().numpy()

    input_np = np.array(input_image).astype(np.uint8)
    resized_image = cv2.resize(input_np, imge_size, interpolation=cv2.INTER_LINEAR)
    return output, resized_image

# =========================================================
# Main UI (Tabs: All images / Single image)
# =========================================================
if st.session_state.uploaded_files:
    tab_all, tab_single = st.tabs([t("tab_all"), t("tab_single")])

    # ---------- Tab 1: All Images Analysis ----------
    with tab_all:
        rows_summary = []
        rows_detail = []

        for uf in st.session_state.uploaded_files:
            # UploadedFile은 스트림이 소모될 수 있으니 바이트로 안전 처리
            file_bytes = uf.getvalue()
            output, _ = infer(model, BytesIO(file_bytes), MODEL_BACKEND)  # output: (C,H,W)
            pred_mask = output.argmax(0)                                   # (H,W)
            probs = softmax_channelwise(output)                            # (C,H,W)

            H, W = pred_mask.shape
            total = H * W if H and W else 0

            # --- 이미지별 present classes & 그들의 평균 confidence ---
            present_classes = []
            avg_conf_list = []

            for cid in range(class_num):
                cmask = (pred_mask == cid)
                ccount = int(cmask.sum())
                if ccount > 0:
                    present_classes.append(cid)
                    avg_conf_list.append(float(probs[cid][cmask].mean()))

            # classes: 1-based 표기 (예: "1, 5, 7")
            classes_str = ", ".join(str(c) for c in present_classes) if present_classes else "-"

            # avg_confidence: present classes의 평균 confidence를 평균낸 값
            avg_conf_over_present = float(np.mean(avg_conf_list)) if avg_conf_list else 0.0

            rows_summary.append({
                t("col_image"): uf.name,
                t("col_classes"): classes_str,
                t("col_avg_conf_over_present"): round(avg_conf_over_present, 3),
            })

            # --- 이미지별 상세 행 생성 ---
            row = { t("col_image"): uf.name }

            # (1) class 1 ~ class N : 픽셀 비율(%)
            for cid in range(class_num):
                pct = (float((pred_mask == cid).sum()) / total * 100.0) if total else 0.0
                col_name = f"{t('cls_col_prefix')}{cid}{t('cls_col_suffix')}"
                row[col_name] = round(pct, 2)

            # (2) class 1 ~ class N avg_confidence : 평균 confidence
            for cid in range(class_num):
                cmask = (pred_mask == cid)
                avgc = float(probs[cid][cmask].mean()) if cmask.any() else 0.0
                col_name = f"{t('cls_col_prefix')}{cid}{t('avg_conf_suffix')}"
                row[col_name] = round(avgc, 3)

            rows_detail.append(row)

        # --- 표 렌더링 ---

        df_summary = pd.DataFrame(rows_summary)
        sum_title_col, sum_btn_col = st.columns([3, 1])
        with sum_title_col:
            st.markdown(t("summary_title_short"))  
        with sum_btn_col:
            csv_sum = df_summary.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=t("download_summary"),       
                data=csv_sum,
                file_name="summary_all_images.csv",
                mime="text/csv",
                key="dl_summary",
                use_container_width=True
            )
        st.dataframe(df_summary, use_container_width=True)


        st.divider()  # 얇은 라인 + 기본 여백
        st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True) 

        
        df_detail = pd.DataFrame(rows_detail)
        dl_title_col, dl_btn_col = st.columns([3, 1])
        with dl_title_col:
            st.markdown(t("detail_title_short"))  
        with dl_btn_col:
            csv_dl = df_detail.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=t("download_detail"),       
                data=csv_dl,
                file_name="detail_per_image.csv",
                mime="text/csv",
                key="dl_detail",
                use_container_width=True
            )
        st.dataframe(df_detail, use_container_width=True)



    # ---------- Tab 2: Single Image Analysis (기존 단일 이미지 뷰) ----------
    with tab_single:
        filenames = [file.name for file in st.session_state.uploaded_files]
        selected_file = st.selectbox(t("pick_image"), filenames)

        for uploaded_file in st.session_state.uploaded_files:
            if uploaded_file.name == selected_file:
                output, input_image = infer(model, uploaded_file, MODEL_BACKEND)

                col1, col2 = st.columns([1, 1], gap="small")

                if view_mode == "class":
                    # 선택 클래스 마스크 / 평균 conf
                    probs = softmax_channelwise(output)
                    prob_map = probs[selected_class]
                    mask_bin = (prob_map > threshold).astype(np.uint8)
                    mask_rgb = np.zeros_like(input_image)
                    color = color_map[selected_class]
                    for i in range(3):
                        mask_rgb[..., i] = mask_bin * color[i]

                    pred_mask = output.argmax(0)
                    cmask = (pred_mask == selected_class)
                    sel_avg_conf = float(prob_map[cmask].mean()) if cmask.any() else 0.0

                    blended = cv2.addWeighted(input_image, 1 - alpha, mask_rgb, alpha, 0)

                    with col1:
                        st.markdown(t("image_title").format(name=uploaded_file.name))
                        image_zoom(blended)
                    with col2:
                        st.metric(label=t("sel_class_metric"), value=f"{sel_avg_conf:.3f}")

                else:
                    # 전체 마스크 색상화 + 비율 & 평균 conf
                    pred_mask = output.argmax(0)
                    mask_rgb = np.zeros_like(input_image)

                    total_pixels = pred_mask.size
                    percentages = {}
                    for cid in range(class_num):
                        class_pixels = int((pred_mask == cid).sum())
                        percentages[cid] = (class_pixels / total_pixels) * 100 if total_pixels else 0.0
                        color = color_map[cid]
                        for i in range(3):
                            mask_rgb[..., i][pred_mask == cid] = color[i]

                    probs = softmax_channelwise(output)
                    mean_conf = {}
                    for cid in range(class_num):
                        cmask = (pred_mask == cid)
                        mean_conf[cid] = float(probs[cid][cmask].mean()) if cmask.any() else 0.0

                    blended = cv2.addWeighted(input_image, 1 - alpha, mask_rgb, alpha, 0)

                    with col1:
                        st.markdown(t("image_title").format(name=uploaded_file.name))
                        image_zoom(blended)
                    with col2:
                        class_ids = list(percentages.keys())
                        df_percent = pd.DataFrame({
                            t("col_class_id"): class_ids,
                            t("col_pixel_pct"): [round(percentages[cid], 2) for cid in class_ids],
                            t("col_avg_conf"): [round(mean_conf[cid], 3) for cid in class_ids],
                        })
                        st.markdown(t("table_title"))
                        st.dataframe(df_percent, use_container_width=True)

                # 마스크 저장 버튼 (단일 이미지)
                mask_save = Image.fromarray(mask_rgb)
                buf = BytesIO()
                mask_save.save(buf, format="PNG")
                buf.seek(0)
                base, _ = os.path.splitext(uploaded_file.name)
                st.download_button(
                    label=t("download_mask"),
                    data=buf,
                    file_name=f"{base}_mask.png",
                    mime="image/png",
                )
                break
