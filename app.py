import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# ------------------------- #
# Streamlit Configuration
# ------------------------- #
st.set_page_config(page_title="Lead Conversion Predictor", page_icon="🤖", layout="centered")
st.title("🤖 Lead Conversion Prediction App")

# Add helper text explaining the two-step process
st.markdown("""
This app predicts lead conversion using a two-step process:

1️⃣ **Train the Model** (Main Area):
   - Upload your CSV file or use the default `Leads.csv`
   - Click "Train Model" to build the predictor
   - CSV must include a `Converted` column (0/1 values)

2️⃣ **Make Predictions** (Sidebar):
   - Fill in lead details using the form
   - Add custom values if needed
   - Click "Submit & Predict" to see results
""")

# ------------------------- #
# Preprocessing + Model Training
# ------------------------- #
def preprocess_and_train(df):
    df = df.copy()

    # Drop columns with unique values (like IDs)
    for col in df.columns:
        if df[col].nunique() == len(df):
            df.drop(col, axis=1, inplace=True)

    # We'll persist encoders/mappings per categorical column so we can reuse them
    encoders = {}

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() <= 10:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
                encoders[col] = {"type": "label", "encoder": le, "encoded_name": col + "_encoded"}
            else:
                # Frequency encoding for high-cardinality columns
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col + "_encoded"] = df[col].map(freq_map)
                encoders[col] = {"type": "freq", "map": freq_map, "encoded_name": col + "_encoded"}

    # Drop original categorical columns
    df.drop(df.select_dtypes(include="object").columns, axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)

    # Check for target column
    if "Converted" not in df.columns:
        raise ValueError("Target column 'Converted' not found in the dataset!")

    # Split features/target
    X = df.drop("Converted", axis=1)
    y = df["Converted"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train model
    model = LogisticRegression(max_iter=100000)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return model, X, accuracy, encoders

# ------------------------- #
# File Upload + Model Train
# ------------------------- #

# Use session state to store trained model and data
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.trained_X = None
    st.session_state.trained_accuracy = None
    st.session_state.trained_encoders = None
    st.session_state.trained_df = None

# Form for data upload and training
with st.form("training_form"):
    uploaded_file = st.file_uploader("📤 Upload a CSV file (must contain a 'Converted' column)", type=["csv"])
    train_submitted = st.form_submit_button("📚 Train Model")

if train_submitted:
    if uploaded_file is not None:
        try:
            # Make sure the file pointer is at the start (UploadedFile may have been read)
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            # Read uploaded CSV and handle empty-data specifically
            try:
                df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.warning("⚠️ Uploaded CSV contains no data or no columns. Falling back to a small demo dataset for the UI.")
                df = pd.DataFrame({
                    "Age": [25, 40, 30],
                    "Income": [50000, 80000, 60000],
                    "State": ["CA", "TX", "NY"],
                    "Converted": [1, 0, 1]
                })

            if df.empty:
                st.warning("⚠️ Uploaded CSV is empty. Falling back to a small demo dataset for the UI.")
                df = pd.DataFrame({
                    "Age": [25, 40, 30],
                    "Income": [50000, 80000, 60000],
                    "State": ["CA", "TX", "NY"],
                    "Converted": [1, 0, 1]
                })

            model, X, accuracy, encoders = preprocess_and_train(df)
            # Store in session state
            st.session_state.trained_model = model
            st.session_state.trained_X = X
            st.session_state.trained_accuracy = accuracy
            st.session_state.trained_encoders = encoders
            st.session_state.trained_df = df.copy()
            st.success(f"✅ Model retrained successfully with uploaded data. Accuracy: **{accuracy*100:.2f}%**")
        except Exception as e:
            st.error(f"❌ Error while processing file: {e}")
            st.stop()
    else:
        try:
            try:
                df = pd.read_csv("Leads.csv")
            except pd.errors.EmptyDataError:
                st.warning("⚠️ Default Leads.csv is empty or invalid. Falling back to a small demo dataset for the UI.")
                df = pd.DataFrame({
                    "Age": [25, 40, 30],
                    "Income": [50000, 80000, 60000],
                    "State": ["CA", "TX", "NY"],
                    "Converted": [1, 0, 1]
                })

            if df.empty:
                st.warning("⚠️ Default Leads.csv is empty. Falling back to a small demo dataset for the UI.")
                df = pd.DataFrame({
                    "Age": [25, 40, 30],
                    "Income": [50000, 80000, 60000],
                    "State": ["CA", "TX", "NY"],
                    "Converted": [1, 0, 1]
                })

            model, X, accuracy, encoders = preprocess_and_train(df)
            # Store in session state
            st.session_state.trained_model = model
            st.session_state.trained_X = X
            st.session_state.trained_accuracy = accuracy
            st.session_state.trained_encoders = encoders
            st.session_state.trained_df = df.copy()
            st.info("📘 Using default Leads.csv dataset")
            st.success(f"✅ Default model trained. Accuracy: **{accuracy*100:.2f}%**")
        except FileNotFoundError:
            st.warning("⚠️ No file uploaded and default Leads.csv not found. Using a small demo dataset so the UI remains interactive.")
            df = pd.DataFrame({
                "Age": [25, 40, 30],
                "Income": [50000, 80000, 60000],
                "State": ["CA", "TX", "NY"],
                "Converted": [1, 0, 1]
            })
            try:
                model, X, accuracy, encoders = preprocess_and_train(df)
                # Store in session state
                st.session_state.trained_model = model
                st.session_state.trained_X = X
                st.session_state.trained_accuracy = accuracy
                st.session_state.trained_encoders = encoders
                st.session_state.trained_df = df.copy()
                st.info("📘 Using built-in demo dataset")
                st.success(f"✅ Demo model trained. Accuracy: **{accuracy*100:.2f}%**")
            except Exception as e:
                st.error(f"❌ Error while training on demo dataset: {e}")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error while loading default dataset: {e}")
            st.stop()

# Ensure prediction variables exist in all code paths
predict_submitted = False
encoded_input = None
input_df = None
user_input = {}

# Only show prediction form if we have a trained model
if st.session_state.trained_model is not None:

    # Initialize prediction variables so they exist even if the form wasn't shown
    predict_submitted = False
    encoded_input = None
    input_df = None
    user_input = {}

# ------------------------- #
# User Input Section (using a form so the app doesn't process until user submits)
# ------------------------- #
    st.sidebar.header("🧩 Enter Lead Features")

    # Get the trained DataFrame from session state
    original_df = st.session_state.trained_df

    # Separate numeric and categorical columns (excluding target)
    numeric_cols = original_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = original_df.select_dtypes(exclude=np.number).columns.tolist()

    if "Converted" in numeric_cols:
        numeric_cols.remove("Converted")
    if "Converted" in categorical_cols:
        categorical_cols.remove("Converted")

    with st.sidebar.form("input_form"):
        user_input = {}

        # Dropdowns for categorical features
        for col in categorical_cols:
            unique_vals = sorted(original_df[col].dropna().unique().tolist())
            # Offer an option at the start to let the user provide a custom value
            add_custom_label = "+ Add custom value"
            options = [add_custom_label] + unique_vals
            choice = st.selectbox(f"{col}", options=options)
            if choice == add_custom_label:
                custom_val = st.text_input(f"Enter custom value for {col}")
                user_input[col] = custom_val if custom_val != "" else None
            else:
                user_input[col] = choice

        # Sliders for numeric features
        for col in numeric_cols:
            mean_val = float(original_df[col].mean())
            min_val = float(original_df[col].min())
            max_val = float(original_df[col].max())

            if min_val == max_val:
                user_input[col] = min_val
                st.info(f"{col} has a constant value of {min_val}")
            else:
                user_input[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

        predict_submitted = st.form_submit_button("🔍 Submit & Predict")

    if predict_submitted:
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([user_input])

        # ------------------------- #
        # Encode user input using persisted encoders from session state
        # ------------------------- #
        encoded_input = input_df.copy()

        for col in encoded_input.columns:
            # Only handle columns that were present in training encoders
            if col in st.session_state.trained_encoders:
                e = st.session_state.trained_encoders[col]
                if e["type"] == "label":
                    le = e["encoder"]
                    val = encoded_input.at[0, col]
                    if pd.isna(val) or val is None or val == "":
                        encoded_input[col] = 0
                    else:
                        try:
                            encoded_input[col] = le.transform([str(val)])[0]
                        except ValueError:
                            # unseen label -> assign new code at end
                            encoded_input[col] = len(le.classes_)
                    # rename to encoded column name
                    encoded_input.rename(columns={col: e["encoded_name"]}, inplace=True)
                elif e["type"] == "freq":
                    freq_map = e["map"]
                    encoded_input[col] = encoded_input[col].map(freq_map).fillna(0)
                    encoded_input.rename(columns={col: e["encoded_name"]}, inplace=True)
            else:
                # Column wasn't categorical during training; keep as-is
                pass

        # Match columns to trained X
        encoded_input = encoded_input.reindex(columns=st.session_state.trained_X.columns, fill_value=0)

    # Add reset button to clear trained model
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Reset Model"):
        # Clear all trained data from session state
        st.session_state.trained_model = None
        st.session_state.trained_X = None
        st.session_state.trained_accuracy = None
        st.session_state.trained_encoders = None
        st.session_state.trained_df = None
        st.rerun()  # Rerun the app to clear the UI

# ------------------------- #
# Prediction Results Section
# ------------------------- #
st.markdown("---")  # Visual separator
with st.container():
    st.subheader("📈 Prediction Results")
    if predict_submitted:
        try:
            prediction = st.session_state.trained_model.predict(encoded_input)[0]
            probability = st.session_state.trained_model.predict_proba(encoded_input)[0][1]
            
            # Show results in columns for better layout
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Prediction:** {'✅ Yes' if prediction == 1 else '❌ No'}")
            with col2:
                st.markdown(f"**Probability:** `{probability*100:.2f}%`")

            # Gauge chart for visual effect
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Conversion Probability (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green" if probability > 0.5 else "red"},
                       'steps': [{'range': [0, 50], 'color': "#ffcccc"},
                                {'range': [50, 100], 'color': "#ccffcc"}]}))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.info("👈 Fill the form in the sidebar and click 'Submit & Predict' to see prediction results.")

# ------------------------- #
# Data Preview
# ------------------------- #
if st.session_state.trained_X is not None:
    with st.expander("📂 Preview Training Data"):
        st.dataframe(st.session_state.trained_X.head())

st.caption("Built with ❤️ using Streamlit and Scikit-learn.")
