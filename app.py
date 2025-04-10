import streamlit as st
import pandas as pd
import statsmodels.api as sm
import joblib

st.set_page_config(page_title="BIA Disease Severity GUI", layout="wide")
st.title("🔬 BIA-Based Disease Severity Prediction GUI")

# Sidebar Upload
st.sidebar.header("Upload Patient BIA Data File")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Make sure your model file is in the same folder

model = load_model()

# Main logic
if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    # ANOVA section
    st.header("📈 ANOVA: Find Important Factors")
    disease_column = st.selectbox("Select Disease Column (target)", df.columns)

    if st.button("Run ANOVA"):
        try:
            independent_vars = df.columns.difference([disease_column])
            formula = f"{disease_column} ~ " + " + ".join(independent_vars)
            model_anova = sm.formula.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model_anova, typ=2)
            st.success("ANOVA Completed!")
            st.dataframe(anova_table)
        except Exception as e:
            st.error(f"Error in ANOVA: {e}")

    # Composition info
    st.header("📋 Disease Composition Information")
    if st.button("Show Disease Composition"):
        st.info("General understanding of parameter impact:")
        st.markdown("""
        - **Obesity**: ↑ Fat %, ↓ Muscle Mass  
        - **Malnutrition**: ↓ Protein, ↓ BMI  
        - **Chronic Kidney Disease (CKD)**: ↑ ECW/TBW ratio, altered impedance
        """)

    # Predict severity from input
    st.header("🧪 Predict Severity for New Patient")
    with st.form("input_form"):
        st.write("Enter the following patient values:")
        input_data = {}
        for col in model.feature_names_in_:
            input_data[col] = st.number_input(f"{col}", value=0.0)
        submitted = st.form_submit_button("Predict Severity")

        if submitted:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Severity: {prediction:.2f}")
