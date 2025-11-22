import streamlit as st
import numpy as np
import json
import ast
from pathlib import Path
from PIL import Image
import os
# reduce TF startup logs (set before importing TF)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# robust import: try tensorflow.keras first, fallback to standalone keras
try:
    from tensorflow.keras.models import load_model
    _ML_BACKEND = "tensorflow"
except Exception:
    try:
        from keras.models import load_model
        _ML_BACKEND = "keras"
    except Exception as e:
        # Streamlit is already imported earlier in the file; show clear message and stop.
        import streamlit as st
        st.error(
            "Could not import TensorFlow or Keras. "
            "Install one of them in the active Python environment:\n\n"
            "  python -m pip install tensorflow\n"
            "or\n"
            "  python -m pip install keras\n"
        )
        # raise to prevent running the app further
        raise ImportError("Missing TensorFlow/Keras") from e

# PAGE SETTINGS
st.set_page_config(page_title="Traffic Sign Classifier", layout="wide")

# PROJECT DIRECTORIES
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = ["model.keras", "traffic_signal_model.h5", "traffic_signal_model.keras", "model.h5"]

# DEFAULT LABELS
DEFAULT_LABELS = [
 "Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)",
 "Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)",
 "End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)",
 "No passing","No passing for vehicles over 3.5 metric tons",
 "Right-of-way at the next intersection","Priority road","Yield","Stop",
 "No vehicles","Vehicles over 3.5 metric tons prohibited","No entry",
 "General caution","Dangerous curve to the left","Dangerous curve to the right",
 "Double curve","Bumpy road","Slippery road","Road narrows on the right",
 "Road work","Traffic signals","Pedestrians","Children crossing",
 "Bicycles crossing","Beware of ice/snow","Wild animals crossing",
 "End of all speed and passing limits","Turn right ahead","Turn left ahead",
 "Ahead only","Go straight or right","Go straight or left","Keep right",
 "Keep left","Roundabout mandatory","End of no passing",
 "End of no passing by vehicles over 3.5 metric tons"
]

# ------------------ BACKEND ------------------

@st.cache_resource
def load_cnn():
    for name in MODEL_CANDIDATES:
        p = PROJECT_DIR / name
        if p.exists():
            try:
                return load_model(str(p)), str(p)
            except Exception as e:
                st.warning(f"Found model file {p} but failed to load: {e}")
    return None, None

@st.cache_resource
def load_or_repair_classes(path="classes.json"):
    p = PROJECT_DIR / path
    if not p.exists():
        with open(p, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_LABELS, f, indent=2, ensure_ascii=False)
        return {str(i): v for i, v in enumerate(DEFAULT_LABELS)}, str(p)

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
            data = ast.literal_eval(raw)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_LABELS, f, indent=2, ensure_ascii=False)
            data = DEFAULT_LABELS

    if isinstance(data, list):
        return {str(i): v for i, v in enumerate(data)}, str(p)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}, str(p)

    return {str(i): v for i, v in enumerate(DEFAULT_LABELS)}, str(p)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.array(img).astype("float32")
    arr = np.sum(arr / 3.0, axis=2, keepdims=True)
    arr = (arr - 128.0) / 128.0
    return arr.reshape(1, 32, 32, 1)

model, model_path = load_cnn()
classes_map, classes_path = load_or_repair_classes()

# ------------------ MODERN UI ------------------

st.markdown("""
<style>
    body { background-color: #0d1117; }
    .title {
        font-size: 40px;
        font-weight: 900;
        text-align: center;
        color: #34d1ff;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #cfcfcf;
        margin-bottom: 25px;
        font-size: 16px;
    }
    .prediction-box {
        background-color: #002f47;
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 0 10px rgba(0,255,255,0.2);
    }
    .footer {
        text-align: center;
        color: #777;
        font-size: 12px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>🚦 Traffic Sign Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a traffic sign image for instant AI classification</div>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2], gap="large")

# ⭐ FIXED: No wrapper -> No empty box
with col1:
    st.markdown("### Upload Image")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    #uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)

    with col1:
        st.markdown("### Uploaded Image")
        st.image(img, caption="Preview", width=180, clamp=True)

    if model is None:
        st.error("❌ AI Model not found.")
    else:
        x = preprocess(img)
        preds = model.predict(x)[0]

        top_idxs = preds.argsort()[::-1][:5].astype(int)
        top_probs = [float(preds[i]) for i in top_idxs]
        top_names = [classes_map.get(str(i), f"Class {i}") for i in top_idxs]

        with col2:
            st.markdown(f"""
                <div class="prediction-box">
                    <h4 style="margin-bottom:5px;">Prediction Result</h4>
                    <h2 style="margin-top:0;"><b>{top_names[0]}</b></h2>
                    <p>Confidence: {top_probs[0]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔍 Top Predictions")
            for name, prob in zip(top_names, top_probs):
                st.write(f"**{name} — {prob*100:.2f}%**")
                st.progress(prob)

else:
    st.info("📤 Upload an image to classify.")

st.markdown("<div class='footer'>Powered by AI Traffic Sign Recognition</div>", unsafe_allow_html=True)
