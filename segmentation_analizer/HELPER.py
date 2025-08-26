import streamlit as st
import pandas as pd

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="HELPER", page_icon="👋", layout="wide")

# ------------------------
# Language state + buttons
# ------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "ko"  # default: Korean

# Language switch buttons
lang_toggle = st.toggle("Korean / English", value=(st.session_state.lang == "ko"))
st.session_state.lang = "ko" if lang_toggle else "en"
# ------------------------
# Localized content
# ------------------------
if st.session_state.lang == "ko":
    st.write("# 🧰 HELPER — 모델 분석 도우미")

    st.markdown(
        """
이미지 세그멘테이션 결과를 빠르게 확인·비교·요약하는 분석 툴입니다.  
여러 장의 이미지를 한 번에 업로드해 **전체 요약(집계)**과 **단일 이미지 상세**를 오가며 확인하고, 결과를 **CSV/PNG**로 저장할 수 있습니다.
    
    
    
"""
    )




    st.markdown(
        """
## 🚀 주요 기능
#### **Segmentation (결과 분석)**

**1) 전체 이미지 요약**  
- 업로드된 모든 이미지에 대해 **클래스별 점유율/평균 confidence** 집계  
- **요약 테이블**과 **이미지별 상세 테이블** 제공  
- 각 테이블을 **CSV로 다운로드**

**2) 단일 이미지 분석**  
- 세그멘테이션 결과(클래스 별색) **오버레이 시각화**  
- **클래스별 픽셀 비율(%)**, **평균 confidence** 표 제공  
- **선택 클래스 마스크 뷰**(threshold 적용) + 평균 confidence 지표  
- 결과 **마스크(PNG) 다운로드** 및 **이미지 확대(zoom)**

**3) 부가 기능**  
- **다국어 UI**: 한국어/영어 전환 지원  
- **모델/런타임 자동 처리**: ONNX Runtime 우선 → 실패 시 PyTorch DeepLabV3-ResNet101 자동 다운로드/로딩(폴백)
"""
    )

    data = {
        "항목": [
            "Confidence Threshold", "Mask Opacity", "View Mode", "#Classes",
            "Select Class", "Language"
        ],
        "라벨": [
            "컨피던스 임계값",
            "마스크 불투명도",
            "보기 모드",
            "클래스 개수",
            "클래스 선택",
            "언어",
        ],
        "타입/범위": [
            "float [0~1]",
            "float [0~1]",
            "all / class",
            "int [1~512]",
            "int [0~K-1]",
            "ko / en",
        ],
        "기본값": [
            "0.5",
            "0.3",
            "all",
            "자동 추정 또는 21",
            "0",
            "ko",
        ],
        "설명": [
            "클래스별 마스크 모드에서 선택 클래스 확률맵 p > t 기준 이진화",
            "원본과 마스크 오버레이 비율",
            "all: argmax 전체 색상화, class: 선택 클래스만 표시",
            "모델 출력 채널에서 가능하면 자동, 실패 시 수동 입력",
            "클래스별 마스크 모드에서 사용할 클래스",
            "UI 텍스트 전환(한국어/영어)",
        ],
    }
    df = pd.DataFrame(data)

else:
    st.write("# 🧰 HELPER — Model Analysis Helper")
    st.markdown(
        """
A tool to quickly **visualize, analyze, and summarize** image segmentation results.  
Upload multiple images to switch between **aggregated summary** and **per-image details**, and export results as **CSV/PNG**.
"""
    )

    st.markdown(
        """
## 🚀 Key Features
#### **Segmentation (Result Analysis)**

**1) All Images Summary**  
- Aggregate **class-wise share / average confidence** across all uploads  
- Provide **summary table** and **per-image detailed table**  
- Download each table as **CSV**

**2) Single Image Analysis**  
- **Overlay visualization** with class-wise colors  
- **Class-wise pixel ratio(%)** and **average confidence** table  
- **Class-mask view** (thresholded) + average confidence metric  
- Download **mask (PNG)** and **zoom** into the image

**3) Extras**  
- **Bilingual UI**: Korean / English  
- **Auto backend**: Prefer ONNX Runtime → fallback to PyTorch DeepLabV3-ResNet101 on failure
"""
    )

    data = {
        "Param": [
            "Confidence Threshold", "Mask Opacity", "View Mode", "#Classes",
            "Select Class", "Language"
        ],
        "Label" : [
            "Confidence Threshold",
            "Mask Opacity",
            "View Mode",
            "Classes",
            "Select Class",
            "Language",
        ],
        "Type/Range": [
            "float [0~1]",
            "float [0~1]",
            "all / class",
            "int [1~512]",
            "int [0~K-1]",
            "ko / en",
        ],
        "Default": [
            "0.5",
            "0.3",
            "all",
            "Auto or 21",
            "0",
            "ko",
        ],
        "Description": [
            "In class-mask mode, binarize the selected class probability map by p > t.",
            "Blend ratio between the original image and the mask overlay.",
            "all: argmax-based colorization, class: show the selected class only.",
            "Auto-detected from output channels when possible; otherwise manual.",
            "Class to use in class-mask mode.",
            "Toggle UI text (Korean/English).",
        ],
    }
    df = pd.DataFrame(data)

# ------------------------
# Render table
# ------------------------
st.subheader("⚙️ Inference Parameters" if st.session_state.lang == "en" else "⚙️ 추론 파라미터")
st.table(df)
