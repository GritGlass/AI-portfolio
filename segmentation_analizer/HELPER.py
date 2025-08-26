import streamlit as st

st.set_page_config(
    page_title="HELPER",
    page_icon="👋",
)

st.write("# 🧰 HELPER — 모델 분석 도우미")

st.markdown(
    """
    이미지 세그멘테이션 결과를 빠르게 확인·비교·요약하는 분석 툴입니다.    
    여러 장의 이미지를 한 번에 업로드해 **전체 요약(집계)**과 단일 이미지 상세를 오가며 확인하고, 결과를 CSV/PNG로 저장할 수 있습니다.
"""
)

st.markdown(
    """
    ## 🚀 주요 기능
    #### **Segmentation (결과 분석)**

    1) 전체 이미지 요약

    - 업로드된 모든 이미지에 대해 클래스별 점유율/평균 confidence 집계
    - 요약 테이블과 이미지별 상세 테이블 제공
    - 각 테이블을 CSV로 다운로드

    2) 단일 이미지 분석

    - 세그멘테이션 결과(클래스 별색) 오버레이 시각화
    - 클래스별 픽셀 점유율(%), 평균 confidence 표 제공
    - 선택 클래스 마스크 뷰(threshold 적용) + 평균 confidence 지표 표시
    - 결과 마스크(PNG) 다운로드 및 이미지 확대(zoom)
    
    3) 부가 기능
    - 다국어 UI: 한국어/영어 전환 지원
    - 모델/런타임 자동 처리 : 우선 ONNX Runtime 사용 → 실패 시 PyTorch DeepLabV3-ResNet101 자동 다운로드/로딩(폴백)
"""
)

st.markdown(
    """
    ### ⚙️ Inference 파라미터
    | 항목                  | 라벨(ko/en)                          |        타입/범위 |             기본값      | 설명                                          |
    | -------------------- | ------------------------------     - | --------------: | ----------------     : | ------------------------------------------- |
    | Confidence Threshold | 컨피던스 임계값 / Confidence Threshold |   `float [0~1]` |                   `0.5`| **클래스별 마스크** 모드에서 선택 클래스 확률맵 `p > t` 기준 이진화 |
    | Mask Opacity         | 마스크 불투명도 / Mask Opacity         |   `float [0~1]` |                   `0.3`| 원본과 마스크 오버레이 비율                             |
    | View Mode            | 보기 모드 / View Mode                 | `all` / `class` |                   `all`| `all`: argmax 전체 색상화, `class`: 선택 클래스만 표시   |
    | #Classes             | 클래스 개수 / #Classes                |   `int [1~512]` |   **자동 추정** 또는 `21`| 모델 출력 채널에서 가능하면 자동, 실패 시 수동 입력              |
    | Select Class         | 클래스 선택 / Select Class            |   `int [0~K-1]` |                     `0`| 클래스별 마스크 모드에서 사용할 클래스                       |
    | Image Size           | 내부 해상도                           |         `(H,W)` |             `(224,224)`| 전처리 `Resize`에 사용(모델 요구에 맞게 조절)              |
    | Backend              | 백엔드                               |  `onnx`/`torch` |                     자동| ONNX 로드 성공 시 onnxruntime, 실패 시 PyTorch로 폴백  |
    | Provider             | 실행 장치                             |        CPU/CUDA |                환경 의존| CUDA 사용 시 `onnxruntime-gpu` 및 드라이버 호환 필요    |
    | Language             | 언어                                 |           ko/en |                    `ko`| UI 텍스트 전환(한국어/영어)                           |

        
"""
)
