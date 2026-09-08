import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_random_forest_pipeline.pkl")

st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="",
    layout="wide"
)

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

div.stButton > button:hover {
    background-color: #125451;
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

st.title("E-Commerce Customer Churn Predictor")
st.write("Estimate the probability that a customer may churn based on their behaviour and profile.")

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
            [
                "Laptop & Accessory",
                "Mobile Phone",
                "Fashion",
                "Mobile",
                "Grocery",
                "Others"
            ]
        )

        payment_mode = st.selectbox(
            "Preferred Payment Mode",
            [
                "Debit Card",
                "Credit Card",
                "E wallet",
                "UPI",
                "Cash on Delivery"
            ]
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
            max_value=127.0,
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

    predict = st.button("Predict Churn")

    if predict:

        input_data = pd.DataFrame({
            "Tenure": [tenure],
            "PreferredLoginDevice": ["Mobile Phone"],
            "CityTier": [1],
            "WarehouseToHome": [warehouse_distance],
            "PreferredPaymentMode": [payment_mode],
            "Gender": ["Male"],
            "HourSpendOnApp": [3.0],
            "NumberOfDeviceRegistered": [4],
            "PreferedOrderCat": [order_category],
            "SatisfactionScore": [satisfaction],
            "MaritalStatus": ["Married"],
            "NumberOfAddress": [3],
            "Complain": [1 if complaint == "Yes" else 0],
            "OrderAmountHikeFromlastYear": [15.0],
            "CouponUsed": [1.0],
            "OrderCount": [2.0],
            "DaySinceLastOrder": [days_last_order],
            "CashbackAmount": [163.0]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        st.write("### Prediction Result")

        if prediction == 1:

            st.markdown("""
            <div class="high-risk">
                <h3 style="color:#9f2d2d !important;">HIGH CHURN RISK</h3>
                <p style="color:#5f1f1f !important;">
                This customer is predicted to have a high probability of churn.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.write(
                "Recommended action: Prioritize this customer for retention "
                "outreach and targeted engagement."
            )

        else:

            st.markdown("""
            <div class="low-risk">
                <h3 style="color:#176b55 !important;">LOW CHURN RISK</h3>
                <p style="color:#245746 !important;">
                This customer is predicted to have a lower probability of churn.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.write(
                "Recommended action: Continue regular engagement and monitor "
                "future customer activity."
            )

        st.progress(float(probability))


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
            max_value=127.0,
            value=15.0
        )

        payment_mode = st.selectbox(
            "Preferred Payment Mode",
            [
                "Debit Card",
                "Credit Card",
                "E wallet",
                "UPI",
                "Cash on Delivery"
            ]
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
            [
                "Laptop & Accessory",
                "Mobile Phone",
                "Fashion",
                "Mobile",
                "Grocery",
                "Others"
            ]
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

    predict = st.button("Predict Churn")

    if predict:

        input_data = pd.DataFrame({
            "Tenure": [tenure],
            "PreferredLoginDevice": [login_device],
            "CityTier": [city_tier],
            "WarehouseToHome": [warehouse_distance],
            "PreferredPaymentMode": [payment_mode],
            "Gender": [gender],
            "HourSpendOnApp": [app_hours],
            "NumberOfDeviceRegistered": [devices],
            "PreferedOrderCat": [order_category],
            "SatisfactionScore": [satisfaction],
            "MaritalStatus": [marital_status],
            "NumberOfAddress": [number_address],
            "Complain": [1 if complaint == "Yes" else 0],
            "OrderAmountHikeFromlastYear": [order_hike],
            "CouponUsed": [coupon_used],
            "OrderCount": [order_count],
            "DaySinceLastOrder": [days_last_order],
            "CashbackAmount": [cashback]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        st.write("### Prediction Result")

        if prediction == 1:

            st.markdown("""
            <div class="high-risk">
                <h3 style="color:#9f2d2d !important;">HIGH CHURN RISK</h3>
                <p style="color:#5f1f1f !important;">
                This customer is predicted to have a high probability of churn.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.write(
                "Recommended action: Prioritize this customer for retention "
                "outreach and targeted engagement."
            )

        else:

            st.markdown("""
            <div class="low-risk">
                <h3 style="color:#176b55 !important;">LOW CHURN RISK</h3>
                <p style="color:#245746 !important;">
                This customer is predicted to have a lower probability of churn.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.write(
                "Recommended action: Continue regular engagement and monitor "
                "future customer activity."
            )

        st.progress(float(probability))
