import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "mechanical_defect_detector.keras"

st.set_page_config(
    page_title="MechSight AI | Surface Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
  background: radial-gradient(circle at 10% 0%, #132c40 0, #09131f 42%, #07101a 100%);
  color: #eaf2f8;
}
[data-testid="stSidebar"] { background: #0b1724; border-right: 1px solid #203548; }
h1, h2, h3 { letter-spacing: -0.025em; }
.hero {
  padding: 1.4rem 1.5rem; border: 1px solid #23475a; border-radius: 20px;
  background: linear-gradient(120deg, rgba(18,49,67,.95), rgba(12,29,44,.9));
  margin-bottom: 1.2rem;
}
.eyebrow { color: #54e5d0; font-size: .78rem; font-weight: 800; letter-spacing: .16em; }
.muted { color: #9fb3c5; }
.pill { display:inline-block; padding:.25rem .6rem; border-radius:999px;
  border:1px solid #2c5967; color:#8cf5e2; font-size:.78rem; margin-right:.35rem; }
div.stButton > button[kind="primary"] {
  background: #48e0ce; color: #06131c; border: 0; font-weight: 800;
  border-radius: 10px; min-height: 3rem;
}
div.stButton > button[kind="primary"]:hover { background: #8cf5e2; color:#06131c; }
.result {
  padding: 1.1rem 1.25rem; border-radius: 16px; margin: .7rem 0 1rem 0;
  border: 1px solid #31556a; animation: fadein .45s ease-out;
}
@keyframes fadein { from { opacity:0; transform:translateY(8px); } to {opacity:1;transform:translateY(0);} }
div[data-testid="stMetric"] { background: #102132; border: 1px solid #223b4f; padding: 1rem; border-radius: 14px; }
hr { border-color: #23394b; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the trained AI model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Upload it beside app.py in the repository."
        )
    return tf.keras.models.load_model(MODEL_PATH)

with st.sidebar:
    st.markdown("## 🔬 MechSight AI")
    st.caption("MECHANICAL SURFACE INTELLIGENCE")
    page = st.radio("NAVIGATION", ["Inspection Studio", "About the Project"])
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.markdown("🟢 Model-based classification")
    st.caption("MobileNetV2 · Binary image classifier")
    st.divider()
    st.caption("Research / educational prototype. Verify results with physical inspection.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-POWERED QUALITY ASSURANCE</div>
  <h1 style="margin:.35rem 0 .4rem 0;">MechSight AI</h1>
  <p style="font-size:1.08rem;margin-bottom:.9rem;color:#c1d2df;">
    Surface defect classification for mechanical components.
  </p>
  <span class="pill">NORMAL · N</span>
  <span class="pill">PITTING · P</span>
  <span class="pill">MOBILENETV2</span>
</div>
""", unsafe_allow_html=True)

if page == "Inspection Studio":
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("### 01 / Upload component image")
        st.write("Choose a clear RGB image of a component surface. The model evaluates the whole image.")
        uploaded = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Supported formats: PNG, JPG, JPEG, BMP, WEBP"
        )
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Image ready for inspection", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read this image: {exc}")
                image = None
        else:
            image = None
            st.info("Upload an image to enable inspection.")
        inspect = st.button("🔍 Inspect Component", type="primary", use_container_width=True, disabled=image is None)

    with right:
        st.markdown("### 02 / Inspection guide")
        st.markdown("""
**Workflow**
1. Upload a surface image.
2. Select **Inspect Component**.
3. Review the predicted class and both class scores.

**Output classes**
- 🟢 **Normal (N):** the model favours the normal class.
- 🔴 **Defective / Pitting (P):** the model favours the pitting class.
""")
        st.warning("This is whole-image classification. It does not draw a bounding box or pinpoint the exact defect location.")

    if inspect and image is not None:
        try:
            with st.spinner("Analyzing component surface..."):
                model = load_model()
                resized = image.resize((150, 150))
                arr = np.asarray(resized, dtype=np.float32)
                arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                arr = np.expand_dims(arr, axis=0)
                p_defect = float(np.asarray(model.predict(arr, verbose=0)).reshape(-1)[0])
                p_defect = min(max(p_defect, 0.0), 1.0)
                p_normal = 1.0 - p_defect

            is_defective = p_defect >= 0.5
            label = "DEFECTIVE — PITTING CLASS" if is_defective else "NORMAL CLASS"
            confidence = p_defect if is_defective else p_normal
            bg = "#351e26" if is_defective else "#12342f"
            border = "#ff7187" if is_defective else "#51e1c7"
            symbol = "⚠️" if is_defective else "✅"
            st.markdown("### 03 / Inspection result")
            st.markdown(
                f'<div class="result" style="background:{bg};border-color:{border};">'
                f'<div style="font-size:.78rem;letter-spacing:.12em;color:#b9cbd8;">MODEL PREDICTION</div>'
                f'<div style="font-size:1.45rem;font-weight:800;margin-top:.3rem;">{symbol} {label}</div>'
                f'<div style="margin-top:.35rem;color:#d4e1e9;">Predicted-class score: <b>{confidence*100:.2f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Normal score", f"{p_normal*100:.2f}%")
            m2.metric("Defective / pitting score", f"{p_defect*100:.2f}%")
            st.markdown("**Class score comparison**")
            st.write(f"Normal (N) · {p_normal*100:.2f}%")
            st.progress(p_normal)
            st.write(f"Defective / Pitting (P) · {p_defect*100:.2f}%")
            st.progress(p_defect)
            st.caption("Scores are model outputs, not guarantees of component quality. Confirm important findings with a qualified physical inspection.")
            with st.expander("Previously measured model evaluation"):
                a, b, c = st.columns(3)
                a.metric("Held-out test accuracy", "95.81%")
                b.metric("Normal recall", "94.88%")
                c.metric("Defect recall", "96.79%")
                st.caption("These metrics are from the project's earlier held-out test set; they are not calculated from the uploaded image.")
        except Exception as exc:
            st.error(f"Inspection failed: {exc}")
            st.info("Check the deployment logs and confirm that the model file is beside app.py.")

else:
    st.markdown("### About MechSight AI")
    st.write(
        "MechSight AI is an educational AI prototype that classifies mechanical surface images "
        "into normal and pitting/defective classes using a trained convolutional neural network."
    )
    st.markdown("#### Model architecture")
    st.write("MobileNetV2 feature extractor with a custom binary classification head.")
    st.markdown("#### Dataset and evaluation")
    st.write(
        "The project was trained on labelled normal and pitting surface images. "
        "The previously recorded held-out test accuracy was 95.81%."
    )
    st.markdown("#### Current limitation")
    st.info(
        "This system performs image-level classification, not object detection. "
        "It cannot identify the exact pixel location of a defect or produce a validated bounding box."
    )

st.divider()
st.caption("MECHSIGHT AI · AI-assisted inspection prototype · Use as decision support, not as a replacement for physical quality inspection.")
