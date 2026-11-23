
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EcoVision AI - Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = (160, 160)

MODEL_FILES = [
    BASE_DIR / "waste_classification_model.keras",
    BASE_DIR / "best_waste_model.keras",
    BASE_DIR / "models" / "waste_classification_model.keras",
    BASE_DIR / "models" / "best_waste_model.keras"
]

CLASS_FILE = BASE_DIR / "class_names.json"

WASTE_INFO = {
    "cardboard": (
        "📦",
        "Cardboard boxes and packaging.",
        "Flatten boxes and keep them dry before recycling."
    ),
    "glass": (
        "🍾",
        "Glass bottles and jars.",
        "Empty and rinse glass containers before recycling."
    ),
    "metal": (
        "🥫",
        "Metal cans and containers.",
        "Empty and rinse metal cans before recycling."
    ),
    "paper": (
        "📄",
        "Newspapers and paper products.",
        "Keep paper clean and dry before recycling."
    ),
    "plastic": (
        "🧴",
        "Plastic bottles and packaging.",
        "Empty and rinse plastic containers."
    ),
    "trash": (
        "🗑️",
        "General waste.",
        "Check whether the item can be reused or recycled separately."
    )
}

# =====================================================
# 2. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1512, #10251d, #142d22);
    color: #F3F8F4;
}

[data-testid="stSidebar"] {
    background: #101d18;
    border-right: 1px solid #294637;
}

[data-testid="stSidebar"] * {
    color: #E8F3EB;
}

.hero {
    padding: 28px;
    border: 1px solid #315744;
    border-radius: 20px;
    background: linear-gradient(120deg, #193c2b, #12271e);
    margin-bottom: 22px;
}

.hero h1 {
    color: #A8F0B9;
    font-size: 34px;
}

.hero p {
    color: #C6D9CC;
    font-size: 16px;
}

.panel {
    background: #172820;
    border: 1px solid #304b3b;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
}

.number {
    color: #A8F0B9;
    font-size: 28px;
    font-weight: bold;
}

.small-label {
    color: #A8C5B2;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.tip {
    background: #20392a;
    border-left: 4px solid #73D996;
    padding: 15px;
    border-radius: 8px;
    margin-top: 12px;
}

div.stButton > button {
    background: #8DE5A5;
    color: #10251A;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 44px;
}

div.stButton > button:hover {
    background: #B3F4C2;
    color: #10251A;
}

.footer {
    text-align: center;
    color: #8BAA97;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. LOAD CLASS NAMES AND MODEL
# =====================================================

@st.cache_data
def load_classes():
    if not CLASS_FILE.exists():
        raise FileNotFoundError(
            "class_names.json not found in project folder."
        )

    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        classes = json.load(file)

    if not isinstance(classes, list) or len(classes) != 6:
        raise ValueError(
            "class_names.json must contain six class names."
        )

    return [str(name).strip().lower() for name in classes]


def find_model_path():
    for path in MODEL_FILES:
        if path.is_file():
            return path
    return None


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


try:
    CLASS_NAMES = load_classes()
except Exception as error:
    st.error(f"Class file error: {error}")
    st.stop()

MODEL_PATH = find_model_path()

if MODEL_PATH is None:
    st.error("Trained model not found.")
    st.write(
        "Place waste_classification_model.keras beside app.py "
        "or inside the models folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

print("\n========== ECOVISION MODEL INFORMATION ==========")
print("Model file:", MODEL_PATH.name)
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)
print("Class order:", CLASS_NAMES)
print("=================================================\n")

# =====================================================
# 4. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None

# =====================================================
# 5. IMAGE PREPROCESSING AND PREDICTION
# =====================================================

def predict_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.asarray(image, dtype=np.float32)

    # MobileNetV2 preprocessing.
    # This must match the preprocessing used during training.
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    batch = np.expand_dims(array, axis=0)

    expected_shape = model.input_shape

    if isinstance(expected_shape, list):
        expected_shape = expected_shape[0]

    if (
        expected_shape[1] is not None
        and expected_shape[1] != IMG_SIZE[0]
    ) or (
        expected_shape[2] is not None
        and expected_shape[2] != IMG_SIZE[1]
    ):
        raise ValueError(
            f"Model expects {expected_shape}, "
            f"but app uses {IMG_SIZE}."
        )

    raw_scores = np.asarray(
        model.predict(batch, verbose=0)
    ).reshape(-1)

    if len(raw_scores) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(raw_scores)} scores, "
            f"but class_names.json contains {len(CLASS_NAMES)} classes."
        )

    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("Model returned invalid prediction scores.")

    if (
        np.any(raw_scores < 0)
        or not np.isclose(raw_scores.sum(), 1.0, atol=0.05)
    ):
        shifted = raw_scores - np.max(raw_scores)
        exp_values = np.exp(shifted)
        scores = exp_values / exp_values.sum()
    else:
        scores = raw_scores

    best_index = int(np.argmax(scores))
    predicted_class = CLASS_NAMES[best_index]
    confidence = float(scores[best_index]) * 100

    # Debug output in VS Code terminal.
    print("\n========== IMAGE PREDICTION DEBUG ==========")
    print("Image size:", image.size)
    print("Input tensor:", batch.shape)
    print("Class order:", CLASS_NAMES)
    print("Raw outputs:", raw_scores)
    print("Probabilities:", scores)
    print("Predicted category:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")
    print("============================================\n")

    return {
        "category": predicted_class,
        "confidence": confidence,
        "scores": {
            CLASS_NAMES[i]: float(scores[i]) * 100
            for i in range(len(CLASS_NAMES))
        },
        "raw_scores": raw_scores.tolist()
    }

# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## ♻️ EcoVision AI")
    st.caption("Smart Waste Intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🌱 Recycling Guide",
            "🤖 About the AI"
        ]
    )

    st.divider()

    st.write("♻️ Six waste categories")
    st.write("🧠 Trained MobileNetV2 model")
    st.write("📷 Image classification")
    st.write("📊 Prediction analysis")

    st.divider()
    st.caption("Python · TensorFlow · Streamlit")

# =====================================================
# 7. DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="small-label">
            AI · SUSTAINABILITY · DEEP LEARNING
        </div>
        <h1>♻️ Smart Waste Classification</h1>
        <p>
        Upload a waste image and let your trained AI model
        identify its category.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Only two dashboard cards.
    # The Images Analysed counter has been removed.

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="panel">
            <div class="small-label">Supported Categories</div>
            <div class="number">06</div>
            Waste types supported
        </div>
        """, unsafe_allow_html=True)

    with c2:
        latest = (
            st.session_state.history[0]["category"].title()
            if st.session_state.history
            else "None"
        )

        st.markdown(
            f"""
            <div class="panel">
                <div class="small-label">Last Prediction</div>
                <div class="number">{latest}</div>
                Latest result
            </div>
            """,
            unsafe_allow_html=True
        )

    left, right = st.columns([1.1, 0.9], gap="large")

    # UPLOAD SECTION

    with left:

        st.markdown("### 📸 Upload Waste Image")

        uploaded = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:

            try:
                image = Image.open(uploaded).convert("RGB")

                upload_id = f"{uploaded.name}_{uploaded.size}"

                if st.session_state.last_upload != upload_id:
                    st.session_state.result = None
                    st.session_state.last_upload = upload_id

                st.image(
                    image,
                    caption="Uploaded waste image",
                    width=350
                )

                st.caption(
                    "Tip: use one clearly visible item with good lighting."
                )

                if st.button(
                    "✨ Analyse Image with AI",
                    use_container_width=True
                ):

                    try:
                        with st.spinner("Analysing image with AI..."):
                            result = predict_image(image)

                        result["time"] = datetime.now().strftime("%H:%M:%S")

                        st.session_state.result = result

                        st.session_state.history.insert(
                            0,
                            {
                                "category": result["category"],
                                "confidence": result["confidence"],
                                "time": result["time"]
                            }
                        )

                        st.session_state.history = (
                            st.session_state.history[:10]
                        )

                    except Exception as error:
                        st.error("Prediction failed.")
                        st.exception(error)

            except Exception as error:
                st.error(f"Could not open image: {error}")

    # HOW IT WORKS SECTION

    with right:

        st.markdown("### 🧭 How it works")

        steps = [
            ("01", "Upload", "Choose a photo of a waste item."),
            ("02", "Preprocess", "Resize the image to 160 × 160 pixels."),
            ("03", "Predict", "The trained model produces six output scores."),
            ("04", "Recycle", "Read guidance for the predicted category.")
        ]

        for number, title, description in steps:

            st.markdown(
                f"""
                <div class="panel">
                    <b style="color:#8DE5A5">
                    {number} · {title}
                    </b>
                    <br>{description}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("""
        <div class="tip">
            🌍 <b>Make every item count</b><br>
            Good waste sorting can help reduce recycling contamination.
            Always follow local recycling rules.
        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # 8. PREDICTION RESULT
    # =================================================

    result = st.session_state.result

    if result is not None:

        category = result["category"]

        icon, description, tip = WASTE_INFO.get(
            category,
            (
                "♻️",
                "Predicted waste category.",
                "Check local recycling rules."
            )
        )

        st.divider()
        st.markdown("## 🔍 AI Analysis Result")

        a, b = st.columns([1, 1.1], gap="large")

        with a:

            st.markdown(
                f"""
                <div class="panel">
                    <div style="font-size:48px">{icon}</div>
                    <div class="small-label">Predicted Category</div>
                    <h2>{category.title()}</h2>
                    <p>{description}</p>
                    <div class="small-label">Model Score</div>
                    <div class="number">{result["confidence"]:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if result["confidence"] < 50:
                st.warning(
                    "The model may be uncertain. Try a clearer image."
                )
            else:
                st.info(
                    "This is the model's highest-scoring category. "
                    "The result is not guaranteed to be correct."
                )

            st.markdown(
                f"""
                <div class="tip">
                    ♻️ <b>Recycling Tip</b><br>
                    {tip}
                </div>
                """,
                unsafe_allow_html=True
            )

        with b:

            st.markdown("### 📊 Scores by Category")

            st.caption(
                "Scores show the model's output for each category."
            )

            sorted_scores = sorted(
                result["scores"].items(),
                key=lambda item: item[1],
                reverse=True
            )

            for name, score in sorted_scores:

                st.write(f"**{name.title()} — {score:.2f}%**")

                st.progress(
                    float(np.clip(score / 100, 0, 1))
                )

            with st.expander("Technical Model Information"):

                st.write("Model file:", MODEL_PATH.name)
                st.write("Model input:", model.input_shape)
                st.write("Class order:", CLASS_NAMES)
                st.write("Raw output:", result["raw_scores"])

    # =================================================
    # 9. PREDICTION HISTORY
    # =================================================

    st.divider()
    st.markdown("### 🕘 Recent Predictions")

    if st.session_state.history:

        for item in st.session_state.history[:5]:

            icon = WASTE_INFO.get(
                item["category"],
                ("♻️", "", "")
            )[0]

            st.markdown(
                f"""
                <div class="panel" style="padding:12px 18px">
                    {icon}
                    <b>{item["category"].title()}</b>
                    · {item["confidence"]:.2f}%
                    · {item["time"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("🧹 Clear Prediction History"):

            st.session_state.history = []
            st.session_state.result = None
            st.rerun()

    else:
        st.info("Your prediction history will appear here.")

# =====================================================
# 10. RECYCLING GUIDE
# =====================================================

elif page == "🌱 Recycling Guide":

    st.markdown("""
    <div class="hero">
        <h1>🌱 Recycling Guide</h1>
        <p>
        Explore waste categories and learn how to manage
        recyclable materials responsibly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Select a waste category",
        CLASS_NAMES,
        format_func=lambda x: x.title()
    )

    icon, description, tip = WASTE_INFO.get(
        selected,
        ("♻️", "Waste category.", "Check local recycling rules.")
    )

    st.markdown(
        f"""
        <div class="panel">
            <div style="font-size:48px">{icon}</div>
            <h2>{selected.title()}</h2>
            <p>{description}</p>
            <div class="tip">
                <b>Recycling Tip:</b><br>{tip}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### All Supported Categories")

    columns = st.columns(3)

    for index, category in enumerate(CLASS_NAMES):

        icon, description, tip = WASTE_INFO.get(
            category,
            ("♻️", "Waste category.", "Check local recycling rules.")
        )

        with columns[index % 3]:

            st.markdown(
                f"""
                <div class="panel" style="min-height:150px">
                    <div style="font-size:30px">{icon}</div>
                    <b>{category.title()}</b><br>
                    {description}
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# 11. ABOUT THE AI
# =====================================================

elif page == "🤖 About the AI":

    st.markdown("""
    <div class="hero">
        <h1>🤖 About EcoVision AI</h1>
        <p>
        A deep-learning application for image-based
        waste classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="panel">
            <h3>🐍 Python</h3>
            Main programming language
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
            <h3>🧠 TensorFlow</h3>
            Deep-learning framework
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="panel">
            <h3>🌐 Streamlit</h3>
            Web application interface
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Prediction Workflow")

    st.write("1. The user uploads a waste image.")
    st.write("2. The app converts it to RGB.")
    st.write("3. The image is resized to 160 × 160 pixels.")
    st.write("4. MobileNetV2 preprocessing is applied.")
    st.write("5. The trained model produces six output scores.")
    st.write("6. The application displays the predicted category.")

    st.markdown("### Supported Categories")

    st.write(
        ", ".join(name.title() for name in CLASS_NAMES)
    )

    st.info(
        "Prediction quality depends on training data, "
        "preprocessing, and class ordering."
    )

# =====================================================
# 12. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    ♻️ EcoVision AI · Smart Waste Classification
    <br>
    Built with Python, TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)
