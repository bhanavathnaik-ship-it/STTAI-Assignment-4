import streamlit as st
import pickle
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="UrbanNest Rent Predictor", page_icon="🏠", layout="centered")

st.title("🏠 UrbanNest Analytics")
st.subheader("Dynamic House Rent Prediction Engine")
st.markdown("Fill in the property details below and click **Predict** to get an estimated rent.")
st.divider()

# ── Load model and encoders ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/best_rf_model.pkl', 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['features'], payload['encoders']

@st.cache_resource
def load_encoders():
    with open('models/label_encoders.pkl', 'rb') as f:
        return pickle.load(f)

model, features, encoders = load_model()

# ── Input widgets ─────────────────────────────────────────────────────────────
# Categorical fields → selectbox (reverse-map from LabelEncoder classes)
# Numerical fields   → number_input / slider

col1, col2 = st.columns(2)

input_data = {}

with col1:
    # City
    if 'city' in encoders:
        city = st.selectbox("City", options=list(encoders['city'].classes_))
        input_data['city'] = city
    else:
        input_data['city'] = st.selectbox("City", ["Mumbai", "Pune", "Delhi", "Hisar"])

    # Property type
    if 'property_type' in encoders:
        property_type = st.selectbox("Property Type", options=list(encoders['property_type'].classes_))
        input_data['property_type'] = property_type

    # Status (Ready to Move / Under Construction)
    if 'Status' in encoders:
        status = st.selectbox("Status", options=list(encoders['Status'].classes_))
        input_data['Status'] = status

    # BHK / bedrooms
    if 'bhk' in features:
        input_data['bhk'] = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)

    if 'bedroom' in features:
        input_data['bedroom'] = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)

with col2:
    # Location
    if 'location' in encoders:
        location = st.selectbox("Location", options=list(encoders['location'].classes_))
        input_data['location'] = location

    # Area / size
    if 'area' in features:
        input_data['area'] = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000, step=50)

    if 'size' in features:
        input_data['size'] = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1000, step=50)

    # Bathrooms
    if 'bathroom' in features:
        input_data['bathroom'] = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

    if 'bathrooms' in features:
        input_data['bathrooms'] = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

    # Floor
    if 'floor' in features:
        input_data['floor'] = st.number_input("Floor No.", min_value=0, max_value=50, value=1, step=1)

    # Age / year built
    if 'age' in features:
        input_data['age'] = st.slider("Property Age (years)", min_value=0, max_value=50, value=5)

# Handle any remaining features not yet covered above
for feat in features:
    if feat not in input_data:
        if feat in encoders:
            val = st.selectbox(feat.replace('_', ' ').title(),
                               options=list(encoders[feat].classes_))
            input_data[feat] = val
        else:
            input_data[feat] = st.number_input(feat.replace('_', ' ').title(),
                                                value=0.0)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict Rent", type="primary"):
    # Build feature vector in the exact order the model expects
    row = []
    for feat in features:
        val = input_data.get(feat, 0)
        if feat in encoders:
            val = int(encoders[feat].transform([str(val)])[0])
        row.append(val)

    X = np.array(row).reshape(1, -1)
    prediction = model.predict(X)[0]

    st.success(f"💰 Estimated Monthly Rent: ₹ {prediction:,.0f}")
    st.caption("Prediction made by a Random Forest model trained on UrbanNest Analytics data.")
