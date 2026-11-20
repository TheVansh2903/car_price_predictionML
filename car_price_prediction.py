import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
import streamlit as st


st.set_page_config(
    page_title="CarWorth - Price Estimator",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "xgb_model.json"
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 82% 8%, rgba(198, 227, 255, 0.65), transparent 24rem),
                linear-gradient(135deg, #f7f9fc 0%, #edf3f8 100%);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 18px;
        }
        .brand-mark {
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: #0b63ce;
            color: white;
            font-size: 22px;
            box-shadow: 0 10px 24px rgba(11,99,206,.20);
        }
        .brand-name {
            font-family: 'Manrope', sans-serif;
            font-size: 21px;
            font-weight: 800;
            color: #10233f;
        }
        .hero-kicker {
            color: #2c66a8;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        .hero-title {
            font-family: 'Manrope', sans-serif;
            color: #10233f;
            font-size: clamp(38px, 5vw, 58px);
            line-height: 1.04;
            letter-spacing: -2.5px;
            margin: 8px 0 12px;
        }
        .hero-copy {
            color: #66758a;
            max-width: 660px;
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 28px;
        }
        [data-testid="stForm"] {
            background: rgba(255,255,255,.94);
            border: 1px solid #e6edf5;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 18px 50px rgba(44,74,111,.10);
        }
        [data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 52px;
            border: 0;
            border-radius: 13px;
            background: #0b63ce;
            color: white;
            font-weight: 700;
            font-size: 16px;
            box-shadow: 0 12px 24px rgba(11,99,206,.20);
        }
        [data-testid="stFormSubmitButton"] button:hover {
            background: #084fA9;
            color: white;
            border: 0;
        }
        .section-label {
            color: #8a98aa;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin: 2px 0 14px;
        }
        .result-card {
            min-height: 390px;
            border-radius: 24px;
            padding: 34px;
            color: white;
            background:
                linear-gradient(145deg, rgba(255,255,255,.10), transparent 35%),
                linear-gradient(135deg, #10233f 0%, #153f70 100%);
            box-shadow: 0 18px 50px rgba(16,35,63,.18);
            position: relative;
            overflow: hidden;
        }
        .result-card:after {
            content: "🚘";
            position: absolute;
            right: -10px;
            bottom: -24px;
            font-size: 150px;
            opacity: .055;
            transform: rotate(-8deg);
        }
        .result-label {
            color: #b8c8dc;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1.1px;
            text-transform: uppercase;
        }
        .result-value {
            font-family: 'Manrope', sans-serif;
            font-size: 52px;
            line-height: 1;
            margin: 18px 0 8px;
            letter-spacing: -2px;
        }
        .result-unit {
            color: #d7e2ef;
            font-size: 16px;
        }
        .result-divider {
            height: 1px;
            background: rgba(255,255,255,.12);
            margin: 30px 0 24px;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            color: #b8c8dc;
            font-size: 13px;
            margin: 12px 0;
        }
        .detail-row strong {
            color: white;
            font-weight: 600;
        }
        .empty-result {
            display: grid;
            align-content: center;
            min-height: 320px;
            text-align: center;
            color: #b8c8dc;
        }
        .empty-icon {
            font-size: 46px;
            margin-bottom: 16px;
            opacity: .8;
        }
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stNumberInput"] > div {
            background: #f8fafc;
            border-radius: 12px;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label {
            color: #40516a;
            font-weight: 600;
        }
        .dataset-note {
            color: #8693a4;
            font-size: 12px;
            margin-top: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🚗</div>
            <div class="brand-name">CarWorth</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-kicker">AI-powered valuation</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Know what your car is worth.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-copy">Get an instant market estimate based on the car’s price, usage, fuel type and ownership history.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        with st.form("valuation_form"):
            st.markdown('<div class="section-label">Vehicle details</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                present_price = st.number_input(
                    "Ex-showroom price",
                    min_value=0.32,
                    max_value=92.60,
                    value=7.63,
                    step=0.10,
                    format="%.2f",
                    help="Original ex-showroom price in lakhs",
                )
                fuel_type = st.selectbox("Fuel type", ("Petrol", "Diesel", "CNG"))
                transmission = st.selectbox("Transmission", ("Manual", "Automatic"))
            with col2:
                kms_driven = st.number_input(
                    "Kilometres driven",
                    min_value=500,
                    max_value=500000,
                    value=36947,
                    step=1000,
                    help="Total distance driven in kilometres",
                )
                seller_type = st.selectbox("Seller type", ("Dealer", "Individual"))
                owners = st.selectbox(
                    "Previous owners",
                    ("First owner", "Second owner", "Third owner", "Fourth or more"),
                    index=0,
                )

            purchase_year = st.selectbox(
                "Purchased year",
                options=list(range(2018, 2002, -1)),
                index=4,
                help="Available dataset range: 2003–2018",
            )

            current_age = datetime.datetime.now().year - purchase_year
            st.markdown(
                f'<div class="dataset-note">Estimated vehicle age: <strong>{current_age} years</strong></div>',
                unsafe_allow_html=True,
            )

            submitted = st.form_submit_button("Estimate selling price")

    with right:
        if "prediction" not in st.session_state:
            st.session_state.prediction = None

        if submitted:
            fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
            seller_map = {"Dealer": 0, "Individual": 1}
            transmission_map = {"Manual": 0, "Automatic": 1}
            owner_map = {
                "First owner": 0,
                "Second owner": 1,
                "Third owner": 2,
                "Fourth or more": 3,
            }

            input_data = pd.DataFrame(
                {
                    "Present_Price": [present_price],
                    "Kms_Driven": [kms_driven],
                    "Fuel_Type": [fuel_map[fuel_type]],
                    "Seller_Type": [seller_map[seller_type]],
                    "Transmission": [transmission_map[transmission]],
                    "Owner": [owner_map[owners]],
                    "Age": [current_age],
                }
            )

            try:
                prediction = float(load_model().predict(input_data)[0])
                st.session_state.prediction = max(prediction, 0)
            except Exception as exc:
                st.error(f"Prediction could not be generated: {exc}")
                st.session_state.prediction = None

        prediction = st.session_state.prediction
        if prediction is not None:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated selling price</div>
                    <div class="result-value">₹{prediction:.2f}</div>
                    <div class="result-unit">lakhs</div>
                    <div class="result-divider"></div>
                    <div class="detail-row"><span>Ex-showroom</span><strong>₹{present_price:.2f} lakh</strong></div>
                    <div class="detail-row"><span>Distance</span><strong>{kms_driven:,.0f} km</strong></div>
                    <div class="detail-row"><span>Vehicle age</span><strong>{current_age} years</strong></div>
                    <div class="detail-row"><span>Fuel / Transmission</span><strong>{fuel_type} / {transmission}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card">
                    <div class="empty-result">
                        <div>
                            <div class="empty-icon">◉</div>
                            <div class="result-label">Your estimate will appear here</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# if __name__ == "__main__":
#     main()

# def main():
#     html_temp = """
#         <h1>Car Price Prediction</h1>
#     """

#     model = xgb.XGBRegressor()
#     model.load_model("xgb_model.json")

#     st.markdown(html_temp, unsafe_allow_html=True)

#     st.markdown("This app will help you to predict your car selling price")

#     p1 = st.number_input("Please enter ex-showroom price (In Lakhs)", 2.5,25.0,step=1.0)

#     p2 = st.number_input("Please enter car driven (In Kilometers)",100,500000,step=100)

#     s1 = st.selectbox("Select the fuel_type",("Petrol","Diesel","CNG"))

#     if s1=='Petrol':
#         p3=0
#     elif s1=='Diesel':
#         p3=1
#     elif s1=='CNG':
#         p3=2

#     s2 = st.selectbox("Select the seller_type",("Dealer","Individual"))

#     if s2=='Dealer':
#         p4=0
#     elif s2=='Individual':
#         p4=1

#     s3 = st.selectbox("Select the transmission",("Manual","Automatic"))

#     if s3=='Manual':
#         p5=0
#     elif s3=='Automatic':
#         p5=1

#     p6 = st.slider("How many owners",0,3)

#     date_time = datetime.datetime.now()
    
#     years = st.number_input("Car purchased year",1990,date_time.year,step=1)

#     p7 = date_time.year - years

#     data_new = pd.DataFrame({
#     'Present_Price':p1,
#     'Kms_Driven':p2,
#     'Fuel_Type':p3,
#     'Seller_Type':p4,
#     'Transmission':p5,
#     'Owner':p6,
#     'Age':p7
# },index=[0])

#     if st.button("Predict"):
#         pred = model.predict(data_new)
#         st.success("You can sell your car at {:.2f} lakhs".format(pred[0]))

# if __name__ == '__main__':
#     main()