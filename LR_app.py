
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="LR Prediction App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
.main {
    padding: 2rem;
}

.stButton>button {
    width: 100%;
    background-color: #3498DB;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 0.5rem;
    border-radius: 10px;
    border: none;
}

.stButton>button:hover {
    background-color: #2C3E50;
}

h1 {
    color: #2C3E50;
    text-align: center;
    padding-bottom: 1rem;
    border-bottom: 3px solid #3498DB;
}
</style>
""", unsafe_allow_html=True)

# ==================== Matplotlib ====================
plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# ==================== 项目路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Final_Model_LR_bundle.pkl")

# ==================== 标题 ====================
st.title("🔬 Logistic Regression Machine Learning Prediction App")
st.markdown("---")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This application uses a Logistic Regression model
    selected through cross-validation and trained on
    selected clinical features.

    **Features:**
    - Real-time prediction
    - Probability estimation
    - SHAP explanation
    """)

    st.header("📋 Instructions")
    st.markdown("""
    1. Enter patient information
    2. Click **Predict**
    3. View prediction probability
    4. Check SHAP explanation
    """)

    st.markdown("---")
    st.caption("Model: L2 Logistic Regression")

# ==================== 标签映射 ====================
LABEL_MAPPINGS = {
    "HB": {"Normal": 1, "Low ": 2},
    "CEA": {"Normal": 1, "High": 2},
    "CA199": {"Normal": 1, "High": 2},
    "CHOL": {"Normal": 1, "High": 2},
    "FOB": {"Negative": 1, "Positive": 2},
    "Number": {"Single": 1, "Multiple": 2},
    "Classification": {
        "Sessile": 1,
        "Semi-pedunculated": 2,
        "Pedunculated": 3
    },
    "Location": {
        "Cecum": 1,
        "Ascending": 2,
        "Transverse": 3,
        "Descending": 4,
        "Sigmoid": 5,
        "Rectum": 6
    },
    "Villous": {"No": 1, "Yes": 2},
    "Erosion": {"No": 1, "Yes": 2},
    "Smooth": {"No": 1, "Yes": 2},
    "Color": {"Normal": 1, "Red": 2, "White": 3}
}

# ==================== 患者输入 ====================
st.header("📝 Patient Information Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographic Information")

    Age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    BMI = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=22.0,
        step=0.1
    )

    Size = st.number_input(
        "Size (cm)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1
    )

with col2:
    st.subheader("Laboratory Tests")

    HB_label = st.selectbox(
        "HB",
        list(LABEL_MAPPINGS["HB"].keys())
    )

    CEA_label = st.selectbox(
        "CEA",
        list(LABEL_MAPPINGS["CEA"].keys())
    )

    CA199_label = st.selectbox(
        "CA199",
        list(LABEL_MAPPINGS["CA199"].keys())
    )

    CHOL_label = st.selectbox(
        "CHOL",
        list(LABEL_MAPPINGS["CHOL"].keys())
    )

    FOB_label = st.selectbox(
        "FOB",
        list(LABEL_MAPPINGS["FOB"].keys())
    )

with col3:
    st.subheader("Clinical Characteristics")

    Number_label = st.selectbox(
        "Number",
        list(LABEL_MAPPINGS["Number"].keys())
    )

    Classification_label = st.selectbox(
        "Classification",
        list(LABEL_MAPPINGS["Classification"].keys())
    )

    Location_label = st.selectbox(
        "Location",
        list(LABEL_MAPPINGS["Location"].keys())
    )

    Villous_label = st.selectbox(
        "Villous Component",
        list(LABEL_MAPPINGS["Villous"].keys())
    )

    Erosion_label = st.selectbox(
        "Erosion",
        list(LABEL_MAPPINGS["Erosion"].keys())
    )

    Smooth_label = st.selectbox(
        "Smooth Surface",
        list(LABEL_MAPPINGS["Smooth"].keys())
    )

    Color_label = st.selectbox(
        "Color",
        list(LABEL_MAPPINGS["Color"].keys())
    )

st.markdown("---")

# ==================== Predict ====================
if st.button("🔮 Predict"):

    try:
        # ==================== 加载模型 ====================
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found:\n{MODEL_PATH}")
            st.stop()

        model_bundle = joblib.load(MODEL_PATH)

        model = model_bundle["model"]
        feature_names = model_bundle["feature_names"]
        feature_space = model_bundle["feature_space"]
        scaler = model_bundle["scaler"]
        decision_threshold = float(
            model_bundle["decision_threshold"]
        )

        # ==================== 标签转数值 ====================
        input_data = {
            "Age": float(Age),
            "BMI": float(BMI),
            "Size": float(Size),

            "HB": int(LABEL_MAPPINGS["HB"][HB_label]),
            "CEA": int(LABEL_MAPPINGS["CEA"][CEA_label]),
            "CA199": int(LABEL_MAPPINGS["CA199"][CA199_label]),
            "CHOL": int(LABEL_MAPPINGS["CHOL"][CHOL_label]),
            "FOB": int(LABEL_MAPPINGS["FOB"][FOB_label]),

            "Number": int(LABEL_MAPPINGS["Number"][Number_label]),
            "Classification": int(
                LABEL_MAPPINGS["Classification"][Classification_label]
            ),
            "Location": int(
                LABEL_MAPPINGS["Location"][Location_label]
            ),
            "Villous": int(
                LABEL_MAPPINGS["Villous"][Villous_label]
            ),
            "Erosion": int(
                LABEL_MAPPINGS["Erosion"][Erosion_label]
            ),
            "Smooth": int(
                LABEL_MAPPINGS["Smooth"][Smooth_label]
            ),
            "Color": int(
                LABEL_MAPPINGS["Color"][Color_label]
            )
        }

        # ==================== 原始数据 ====================
        X_raw = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # ==================== 标准化 ====================
        if feature_space == "selected":

            if scaler is None:
                st.error("❌ Scaler not found.")
                st.stop()

            X_model = pd.DataFrame(
                scaler.transform(X_raw),
                columns=feature_names
            )

        else:
            X_model = X_raw.copy()

        # ==================== 预测 ====================
        prob = model.predict_proba(X_model)[0]

        negative_probability = float(prob[0])
        positive_probability = float(prob[1])

        # 使用训练集确定的决策阈值
        pred = int(
            positive_probability >= decision_threshold
        )

        # ==================== Prediction Results ====================
        st.markdown("---")
        st.header("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction Result",
                "Positive (1)" if pred == 1 else "Negative (0)"
            )

        with col2:
            st.metric(
                "Positive Probability",
                f"{positive_probability:.2%}"
            )

        with col3:
            st.metric(
                "Decision Threshold",
                f"{decision_threshold:.3f}"
            )

        # ==================== Probability Distribution ====================
        st.subheader("Probability Distribution")

        fig, ax = plt.subplots(figsize=(10, 4))

        bars = ax.barh(
            ["Negative (0)", "Positive (1)"],
            [negative_probability, positive_probability],
            color=["#27AE60", "#E74C3C"]
        )

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.grid(
            axis="x",
            alpha=0.3,
            linestyle="--"
        )

        for bar in bars:
            width = bar.get_width()

            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2%}",
                va="center"
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ==================== SHAP ====================
        st.markdown("---")
        st.header("🔍 SHAP Explanation")

        with st.spinner("Computing SHAP values..."):

            # LR使用LinearExplainer
            # StandardScaler标准化后的0对应训练数据均值
            background = np.zeros(
                (1, X_model.shape[1])
            )

            explainer = shap.LinearExplainer(
                model,
                background
            )

            shap_result = explainer(X_model)

            # SHAP值
            shap_values = np.asarray(
                shap_result.values
            )[0]

            # base value
            base_value = shap_result.base_values

            if isinstance(
                base_value,
                (list, np.ndarray)
            ):
                base_value = np.asarray(
                    base_value
                ).reshape(-1)[0]

            # 构建Explanation
            shap_exp = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=X_raw.iloc[0].values,
                feature_names=feature_names
            )

            # ==================== Waterfall ====================
            st.subheader("SHAP Waterfall Plot")

            st.info(
                "This plot shows how each feature contributes "
                "to the prediction for this individual patient."
            )

            fig_shap = plt.figure(
                figsize=(12, 8),
                facecolor="white"
            )

            shap.plots.waterfall(
                shap_exp,
                max_display=15,
                show=False
            )

            plt.tight_layout()

            st.pyplot(fig_shap)
            plt.close(fig_shap)

    except Exception as e:
        st.error(
            f"❌ Prediction failed: {str(e)}"
        )
        st.exception(e)

# ==================== 页脚 ====================
st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#7F8C8D;
    padding:1rem;
">
    <p>
    © 2026 Logistic Regression Prediction System |
    Powered by Streamlit & SHAP
    </p>

    <p style="font-size:12px;">
    ⚠️ This tool is for research purposes only.
    Clinical decisions should be made by qualified medical professionals.
    </p>
</div>
""", unsafe_allow_html=True)