
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from severity import calculate_severity


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

WEIGHTS = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "pothole"
    / "pothole_img416_fixed"
    / "weights"
    / "best.pt"
)

st.set_page_config(
    page_title="Road DetectAI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. PREMIUM DARK THEME
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #eef3ff;
    }

    [data-testid="stSidebar"] {
        background-color: #111a2d;
        border-right: 1px solid #283754;
    }

    .hero {
        background: linear-gradient(120deg, #142544, #19385c);
        border: 1px solid #30476b;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero p {
        color: #c4d4ef;
        margin-bottom: 0;
    }

    .metric-card {
        background: #121d32;
        border: 1px solid #2b3c59;
        border-radius: 14px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: #a9bad7;
        font-size: 14px;
    }

    .metric-value {
        color: #66b5ff;
        font-size: 27px;
        font-weight: bold;
        margin-top: 8px;
    }

    .stButton > button {
        background-color: #2379e8;
        color: white;
        border: none;
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. LOAD MODEL
# ==================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.title("🛣️ Road DetectAI")
    st.caption("AI-powered road inspection")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Model Information",
            "About Project",
        ],
    )

    st.divider()

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    st.caption(
        "Lower confidence may detect more defects "
        "but can also produce more false positives."
    )


# ==================================================
# 5. CHECK MODEL FILE
# ==================================================

if not WEIGHTS.is_file():
    st.error("Trained model file was not found.")

    st.code(str(WEIGHTS))

    st.write(
        "Check that best.pt exists in the expected "
        "folder before running detection."
    )

    st.stop()


# ==================================================
# 6. MODEL INFORMATION PAGE
# ==================================================

if page == "Model Information":
    st.title("🤖 Model Information")

    st.write("Architecture: YOLOv8n")
    st.write("Inference device: CPU")
    st.write("Inference image size: 320")
    st.write("Training dataset: Your configured pothole dataset")

    st.write("Model weights path:")
    st.code(str(WEIGHTS))

    st.success("Model weights file found.")

    st.stop()


# ==================================================
# 7. ABOUT PAGE
# ==================================================

if page == "About Project":
    st.title("About Road DetectAI")

    st.write(
        """
        Road DetectAI is an AI-based road damage
        detection application. It uses a trained YOLOv8
        model to identify defects in uploaded road images.
        """
    )

    st.write(
        """
        The dashboard displays predicted bounding boxes,
        confidence scores, detection counts, and estimated
        severity based on bounding-box area coverage.
        """
    )

    st.warning(
        "Image-area coverage is an estimate. It does not "
        "measure the real physical dimensions of a pothole."
    )

    st.stop()


# ==================================================
# 8. MAIN DASHBOARD
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI-Powered Pothole Detection System</h1>
        <p>
            Detect potholes and road defects using your
            trained YOLOv8 model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 9. LOAD TRAINED MODEL
# ==================================================

try:
    model = load_model(str(WEIGHTS))
except Exception as exc:
    st.error(f"Unable to load the YOLO model: {exc}")
    st.stop()


# ==================================================
# 10. IMAGE UPLOAD
# ==================================================

st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Upload a road image to start detection.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 01 · Upload")
        st.write("Choose an image of a road.")

    with col2:
        st.markdown("### 02 · Detect")
        st.write("Run your trained YOLOv8 model.")

    with col3:
        st.markdown("### 03 · Analyze")
        st.write("Review detections and severity.")

    st.stop()


# ==================================================
# 11. READ IMAGE
# ==================================================

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Unable to read the uploaded image: {exc}")
    st.stop()

image_width, image_height = image.size

left, right = st.columns(2)

with left:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)


# ==================================================
# 12. RUN DETECTION
# ==================================================

if st.button(
    "🚀 Run Road Damage Detection",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Analyzing the image..."):

        try:
            result = model.predict(
                source=image,
                conf=confidence,
                imgsz=320,
                device="cpu",
                verbose=False,
            )[0]

        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.stop()

    boxes = []
    rows = []

    if result.boxes is not None and len(result.boxes) > 0:

        coordinates = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()

        for coords, score, class_id in zip(
            coordinates, scores, classes
        ):

            boxes.append(coords)

            class_name = result.names.get(
                int(class_id),
                str(int(class_id)),
            )

            rows.append(
                {
                    "Class": class_name,
                    "Confidence (%)": round(score * 100, 2),
                    "X1": round(coords[0], 2),
                    "Y1": round(coords[1], 2),
                    "X2": round(coords[2], 2),
                    "Y2": round(coords[3], 2),
                }
            )

    # Calculate image coverage and severity.
    coverage, severity = calculate_severity(
        boxes,
        image_width,
        image_height,
    )

    detection_count = len(rows)

    # Generate the annotated image.
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    annotated_image = Image.fromarray(annotated_rgb)

    # Show the annotated image.
    with right:
        st.subheader("Detected Road Damage")
        st.image(
            annotated_image,
            use_container_width=True,
        )

    st.divider()

    # ==================================================
    # 13. DASHBOARD METRICS
    # ==================================================

    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Total Detections
                </div>
                <div class="metric-value">
                    {detection_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Damage Coverage
                </div>
                <div class="metric-value">
                    {coverage:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Severity
                </div>
                <div class="metric-value">
                    {severity}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Coverage is calculated using the sum of predicted "
        "bounding-box areas divided by image area. Overlapping "
        "boxes may count the same area more than once."
    )

    # ==================================================
    # 14. DETECTION TABLE
    # ==================================================

    st.subheader("🔎 Detection Details")

    if rows:

        dataframe = pd.DataFrame(rows)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_data,
            file_name="road_detection_results.csv",
            mime="text/csv",
        )

    else:

        st.warning(
            "No defects detected at this confidence threshold. "
            "Try another image or a lower confidence value."
        )

    # ==================================================
    # 15. DOWNLOAD ANNOTATED IMAGE
    # ==================================================

    buffer = BytesIO()

    annotated_image.save(
        buffer,
        format="PNG",
    )

    st.download_button(
        "⬇️ Download Annotated Image",
        data=buffer.getvalue(),
        file_name="road_detection_annotated.png",
        mime="image/png",
    )

    st.success("Image analysis completed.")
