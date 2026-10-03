import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="📉",
    layout="wide"
)

MODEL_PATH = "churn_xgboost_pipeline.pkl"

# Risk bands applied to the model's churn score (0 to 1)
LOW_RISK_CUTOFF = 0.30    # below this -> low risk
HIGH_RISK_CUTOFF = 0.50   # this & above -> high risk, in between -> medium

# Column order the trained pipeline expects
FEATURE_ORDER = [
    "Tenure",
    "PreferredLoginDevice",
    "CityTier",
    "WarehouseToHome",
    "PreferredPaymentMode",
    "Gender",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

# Values used for the features that Quick Prediction does not ask for
# (typical values from the training data: medians / most frequent categories)
QUICK_DEFAULTS = {
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 1,
    "Gender": "Male",
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 4,
    "MaritalStatus": "Married",
    "NumberOfAddress": 3,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 1.0,
    "OrderCount": 2.0,
    "CashbackAmount": 163.0,
}

ORDER_CATEGORIES = [
    "Laptop & Accessory",
    "Mobile Phone",
    "Fashion",
    "Mobile",
    "Grocery",
    "Others",
]

PAYMENT_MODES = [
    "Debit Card",
    "Credit Card",
    "E wallet",
    "UPI",
    "Cash on Delivery",
]


@st.cache_resource
def load_model(path):
    # Loaded once and reused across reruns / users
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
except Exception as err:
    st.error(
        f"Could not load the model file '{MODEL_PATH}'. "
        "Check that it is in the same folder as app.py and that the "
        "scikit-learn / xgboost versions in requirements.txt match the "
        "versions used for training."
    )
    st.exception(err)
    st.stop()

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #f4f1ea;
    color: #1f2937;
}

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1 {
    color: #17202a !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
}

h2 {
    color: #17202a !important;
    font-size: 1.7rem !important;
}

h3 {
    color: #263238 !important;
}

p, label, .stMarkdown {
    color: #374151 !important;
}

.stSelectbox label,
.stNumberInput label {
    color: #374151 !important;
    font-weight: 500 !important;
}

[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #9ca3af !important;
}

[data-baseweb="select"] span {
    color: #1f2937 !important;
}

[data-baseweb="select"] input {
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
}

[data-baseweb="select"] input::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
}

[data-baseweb="select"] svg {
    fill: #374151 !important;
}

[data-baseweb="popover"] {
    background-color: #ffffff !important;
}

[data-baseweb="popover"] > div {
    background-color: #ffffff !important;
}

div[role="listbox"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

div[role="option"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

div[role="option"] * {
    color: #1f2937 !important;
}

div[role="option"]:hover {
    background-color: #f3f4f6 !important;
    color: #1f2937 !important;
}

div[role="option"][aria-selected="true"] {
    background-color: #e5e7eb !important;
    color: #111827 !important;
}

div[role="option"][aria-selected="true"] * {
    color: #111827 !important;
}

[data-baseweb="menu"] {
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

[data-baseweb="menu"] * {
    color: #1f2937 !important;
}

.stNumberInput input {
    color: #1f2937 !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #1f2937 !important;
}

.stNumberInput button {
    color: #1f2937 !important;
    background-color: #ffffff !important;
}

.stNumberInput button svg {
    fill: #1f2937 !important;
}

div.stButton > button {
    width: 100%;
    background-color: #176b68;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    font-weight: 600;
}

div.stButton > button p,
div.stButton > button:hover p,
div.stButton > button:focus p {
   color: white !important;
}

.result-card {
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
    background-color: #ffffff;
    border: 1px solid #d6d3cd;
}

.low-risk {
    background-color: #e5f3ed;
    border-left: 6px solid #238b6f;
    padding: 18px;
    border-radius: 8px;
}

.medium-risk {
    background-color: #fdf3dc;
    border-left: 6px solid #d9a21b;
    padding: 18px;
    border-radius: 8px;
}

.high-risk {
    background-color: #f9e5e3;
    border-left: 6px solid #c94c4c;
    padding: 18px;
    border-radius: 8px;
}

.info-box {
    background-color: #e7eef0;
    padding: 16px;
    border-radius: 8px;
    margin-top: 15px;
}

hr {
    border-color: #d6d3cd !important;
}

[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #d6d3cd;
    border-radius: 8px;
}

[data-testid="stMetricValue"] {
    color: #17202a !important;
}

[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_input(values):
    """Turn a dict of feature values into a one-row DataFrame in training order."""
    return pd.DataFrame([values])[FEATURE_ORDER]


def show_result(input_data):
    """Run the pipeline on one customer and display the risk band."""
    probabilities = model.predict_proba(input_data)[0]
    
    # Safely resolve target class index for positive churn label (1)
    if hasattr(model, "classes_"):
        classes_list = list(model.classes_)
        if 1 in classes_list:
            churn_index = classes_list.index(1)
        elif "1" in classes_list:
            churn_index = classes_list.index("1")
        else:
            churn_index = 1 if len(classes_list) > 1 else 0
    else:
        churn_index = 1

    score = float(probabilities[0])

    if score >= HIGH_RISK_CUTOFF:
        css_class = "high-risk"
        title = "HIGH CHURN RISK"
        title_color = "#9f2d2d"
        text_color = "#5f1f1f"
        message = "This customer is flagged as high risk of churning."
        action = (
            "Recommended action: Prioritize this customer for retention "
            "outreach and targeted engagement."
        )
    elif score >= LOW_RISK_CUTOFF:
        css_class = "medium-risk"
        title = "MEDIUM CHURN RISK"
        title_color = "#8a6410"
        text_color = "#5c430c"
        message = "This customer shows some signs of churn risk."
        action = (
            "Recommended action: Add this customer to a watchlist and "
            "consider a light-touch engagement such as a check-in message "
            "or a small incentive."
        )
    else:
        css_class = "low-risk"
        title = "LOW CHURN RISK"
        title_color = "#176b55"
        text_color = "#245746"
        message = "This customer is flagged as lower risk of churning."
        action = (
            "Recommended action: Continue regular engagement and monitor "
            "future customer activity."
        )

    st.divider()
    st.write("### Prediction Result")

    st.markdown(f"""
    <div class="{css_class}">
        <h3 style="color:{title_color} !important;">{title}</h3>
        <p style="color:{text_color} !important;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Churn Risk Score", f"{score * 100:.1f}%")
    st.write(action)
    st.progress(min(max(score, 0.0), 1.0))

    st.caption(
        "The risk score is the model's output on a 0-100% scale; it is not a "
        "calibrated probability. Bands: below "
        f"{LOW_RISK_CUTOFF * 100:.0f}% low, "
        f"{LOW_RISK_CUTOFF * 100:.0f}-{HIGH_RISK_CUTOFF * 100:.0f}% medium, "
        f"{HIGH_RISK_CUTOFF * 100:.0f}% and above high."
    )

    with st.expander("Debug Diagnostic Data"):
        st.write("**Model Classes:**", getattr(model, "classes_", "N/A"))
        st.write("**Raw Probabilities:**", probabilities)
        st.write("**Processed Input Row:**")
        st.dataframe(input_data)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("E-Commerce Customer Churn Predictor")
st.write(
    "Estimate the churn risk of a customer based on their behaviour and profile."
)

st.divider()

prediction_mode = st.radio(
    "Prediction Mode",
    ["Quick Prediction", "Detailed Prediction"],
    horizontal=True
)

if prediction_mode == "Quick Prediction":

    st.write("### Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0.0,
            max_value=61.0,
            value=5.0
        )

        order_category = st.selectbox(
            "Preferred Order Category",
            ORDER_CATEGORIES
        )

        payment_mode = st.selectbox(
            "Preferred Payment Mode",
            PAYMENT_MODES
        )

        satisfaction = st.selectbox(
            "Satisfaction Score",
            [1, 2, 3, 4, 5],
            index=2
        )

    with col2:
        complaint = st.selectbox(
            "Has Customer Complained?",
            ["No", "Yes"]
        )

        warehouse_distance = st.number_input(
            "Warehouse to Home Distance",
            min_value=5.0,
            max_value=36.0,
            value=15.0
        )

        days_last_order = st.number_input(
            "Days Since Last Order",
            min_value=0.0,
            max_value=46.0,
            value=3.0
        )

    st.markdown("""
    <div class="info-box">
    Quick Prediction uses the main customer behaviour indicators.
    Other model features are filled using typical values from the training data.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button("Predict Churn"):

        values = dict(QUICK_DEFAULTS)
        values.update({
            "Tenure": float(tenure),
            "WarehouseToHome": float(warehouse_distance),
            "PreferredPaymentMode": str(payment_mode),
            "PreferedOrderCat": str(order_category),
            "SatisfactionScore": int(satisfaction),
            "Complain": 1 if complaint == "Yes" else 0,
            "DaySinceLastOrder": float(days_last_order),
        })

        show_result(build_input(values))

else:

    st.write("### Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0.0,
            max_value=61.0,
            value=5.0
        )

        login_device = st.selectbox(
            "Preferred Login Device",
            ["Mobile Phone", "Computer", "Phone"]
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        warehouse_distance = st.number_input(
            "Warehouse to Home Distance",
            min_value=5.0,
            max_value=36.0,
            value=15.0
        )

        payment_mode = st.selectbox(
            "Preferred Payment Mode",
            PAYMENT_MODES
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        app_hours = st.number_input(
            "Hours Spent on App",
            min_value=0.0,
            max_value=5.0,
            value=3.0
        )

        devices = st.number_input(
            "Number of Devices Registered",
            min_value=1,
            max_value=6,
            value=4
        )

        order_category = st.selectbox(
            "Preferred Order Category",
            ORDER_CATEGORIES
        )

        satisfaction = st.selectbox(
            "Satisfaction Score",
            [1, 2, 3, 4, 5],
            index=2
        )

    with col3:

        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced"]
        )

        number_address = st.number_input(
            "Number of Addresses",
            min_value=1,
            max_value=22,
            value=3
        )

        complaint = st.selectbox(
            "Has Customer Complained?",
            ["No", "Yes"]
        )

        order_hike = st.number_input(
            "Order Amount Hike From Last Year (%)",
            min_value=11.0,
            max_value=26.0,
            value=15.0
        )

        coupon_used = st.number_input(
            "Coupons Used",
            min_value=0.0,
            max_value=16.0,
            value=1.0
        )

    st.write("### Order Activity")

    col4, col5, col6 = st.columns(3)

    with col4:
        order_count = st.number_input(
            "Order Count",
            min_value=1.0,
            max_value=16.0,
            value=2.0
        )

    with col5:
        days_last_order = st.number_input(
            "Days Since Last Order",
            min_value=0.0,
            max_value=46.0,
            value=3.0
        )

    with col6:
        cashback = st.number_input(
            "Cashback Amount",
            min_value=0.0,
            max_value=325.0,
            value=163.0
        )

    st.write("")

    if st.button("Predict Churn"):

        values = {
            "Tenure": float(tenure),
            "PreferredLoginDevice": str(login_device),
            "CityTier": int(city_tier),
            "WarehouseToHome": float(warehouse_distance),
            "PreferredPaymentMode": str(payment_mode),
            "Gender": str(gender),
            "HourSpendOnApp": float(app_hours),
            "NumberOfDeviceRegistered": int(devices),
            "PreferedOrderCat": str(order_category),
            "SatisfactionScore": int(satisfaction),
            "MaritalStatus": str(marital_status),
            "NumberOfAddress": int(number_address),
            "Complain": 1 if complaint == "Yes" else 0,
            "OrderAmountHikeFromlastYear": float(order_hike),
            "CouponUsed": float(coupon_used),
            "OrderCount": float(order_count),
            "DaySinceLastOrder": float(days_last_order),
            "CashbackAmount": float(cashback),
        }

        show_result(build_input(values))
