import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 0. CSS Loading Function ---
# This function reads the custom CSS file and injects it into the Streamlit app.
def load_css(file_name):
    """Reads the CSS file and injects it into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback if CSS file is missing
        pass

# --- 1. DEFINE FEATURES AND SCALING PARAMS ---
# The 28 features used for prediction, derived from cancer_regression.py
FEATURE_COLUMNS = [
    'avgdeathsperyear', 'target_deathrate', 'incidencerate',
    'medincome', 'popest2015', 'povertypercent', 'studypercap',
    'medianage', 'medianagemale', 'medianagefemale',
    'percentmarried', 'pctnohs18_24', 'pcths18_24',
    'pctbachdeg18_24', 'pcths25_over', 'pctbachdeg25_over',
    'pctemployed16_over', 'pctunemployed16_over', 'pctprivatecoverage',
    'pctempprivcoverage', 'pctpubliccoverage',
    'pctpubliccoveragealone', 'pctwhite', 'pctblack', 'pctasian',
    'pctotherrace', 'pctmarriedhouseholds', 'birthrate'
]

# Dictionary mapping short feature names to their full, descriptive titles
FEATURE_NAMES = {
    'avgdeathsperyear': 'Average Deaths Per Year',
    'target_deathrate': 'Target Death Rate (Mortality per 100k)',
    'incidencerate': 'Cancer Incidence Rate (Cases per 100k)',
    'medincome': 'Median Household Income',
    'popest2015': 'Estimated Population in 2015',
    'povertypercent': 'Poverty Percentage',
    'studypercap': 'Study Per Capita (Clinical Trials)',
    'medianage': 'Median Age of Population',
    'medianagemale': 'Median Age (Male)',
    'medianagefemale': 'Median Age (Female)',
    'percentmarried': 'Percent Married (Population 15+)',
    'pctnohs18_24': 'Percent No High School (Age 18-24)',
    'pcths18_24': 'Percent High School Only (Age 18-24)',
    'pctbachdeg18_24': 'Percent Bachelor\'s Degree+ (Age 18-24)',
    'pcths25_over': 'Percent High School Only (Age 25+)',
    'pctbachdeg25_over': 'Percent Bachelor\'s Degree+ (Age 25+)',
    'pctemployed16_over': 'Percent Employed (Age 16+)',
    'pctunemployed16_over': 'Percent Unemployed (Age 16+)',
    'pctprivatecoverage': 'Percent Private Health Coverage',
    'pctempprivcoverage': 'Percent Employee Private Coverage',
    'pctpubliccoverage': 'Percent Public Health Coverage',
    'pctpubliccoveragealone': 'Percent Public Coverage Alone',
    'pctwhite': 'Percent White Population',
    'pctblack': 'Percent Black Population',
    'pctasian': 'Percent Asian Population',
    'pctotherrace': 'Percent Other Race Population',
    'pctmarriedhouseholds': 'Percent Married Households',
    'birthrate': 'Crude Birth Rate (per 1,000)',
}

# !!! CRITICAL: REPLACE THESE DUMMY VALUES !!!
# You MUST replace these placeholder values with the actual min/max values 
# from your original training data (cancer_reg.csv) to correctly apply
# the MinMaxScaler transformation. Failure to do so will result in inaccurate predictions.
SCALER_PARAMS = {
    'avgdeathsperyear': {'min': 0.0, 'max': 5000.0, 'step': 100.0},
    'target_deathrate': {'min': 100.0, 'max': 300.0, 'step': 1.0},
    'incidencerate': {'min': 300.0, 'max': 600.0, 'step': 5.0},
    'medincome': {'min': 20000.0, 'max': 120000.0, 'step': 1000.0},
    'popest2015': {'min': 0, 'max': 10000000, 'step': 50000},
    'povertypercent': {'min': 0.0, 'max': 50.0, 'step': 0.1},
    'studypercap': {'min': 0.0, 'max': 2000.0, 'step': 10.0},
    'medianage': {'min': 20.0, 'max': 60.0, 'step': 0.1},
    'medianagemale': {'min': 20.0, 'max': 60.0, 'step': 0.1},
    'medianagefemale': {'min': 20.0, 'max': 60.0, 'step': 0.1},
    'percentmarried': {'min': 20.0, 'max': 80.0, 'step': 0.1},
    'pctnohs18_24': {'min': 0.0, 'max': 50.0, 'step': 0.1},
    'pcths18_24': {'min': 20.0, 'max': 80.0, 'step': 0.1},
    'pctbachdeg18_24': {'min': 0.0, 'max': 80.0, 'step': 0.1},
    'pcths25_over': {'min': 20.0, 'max': 80.0, 'step': 0.1},
    'pctbachdeg25_over': {'min': 5.0, 'max': 70.0, 'step': 0.1},
    'pctemployed16_over': {'min': 30.0, 'max': 80.0, 'step': 0.1},
    'pctunemployed16_over': {'min': 0.0, 'max': 20.0, 'step': 0.1},
    'pctprivatecoverage': {'min': 40.0, 'max': 90.0, 'step': 0.1},
    'pctempprivcoverage': {'min': 40.0, 'max': 90.0, 'step': 0.1},
    'pctpubliccoverage': {'min': 10.0, 'max': 60.0, 'step': 0.1},
    'pctpubliccoveragealone': {'min': 5.0, 'max': 60.0, 'step': 0.1},
    'pctwhite': {'min': 0.0, 'max': 100.0, 'step': 0.1},
    'pctblack': {'min': 0.0, 'max': 100.0, 'step': 0.1},
    'pctasian': {'min': 0.0, 'max': 20.0, 'step': 0.1},
    'pctotherrace': {'min': 0.0, 'max': 10.0, 'step': 0.1},
    'pctmarriedhouseholds': {'min': 30.0, 'max': 80.0, 'step': 0.1},
    'birthrate': {'min': 5.0, 'max': 25.0, 'step': 0.1},
}

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the pickled Linear Regression model."""
    try:
        # Load the model from the previously generated pickle file
        with open('Cancer_Regression.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'Cancer_Regression.pkl' not found. Please ensure the model file is accessible.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 3. SCALING FUNCTION ---
def min_max_scale_single_value(value, min_val, max_val):
    """Applies the MinMax scaling formula (X - min) / (max - min)."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

# --- NEW: PREDICTOR PAGE FUNCTION ---
def show_predictor_page(model):
    """Displays the main input form and prediction logic."""
    st.title("County Cancer Incidence Predictor (avganncount)")
    st.markdown("""
    This application uses a Linear Regression model trained on demographic and health indicators 
    to predict the average annual count of cancer cases (`avganncount`) in a region.
    """)
    st.warning("""
    **⚠️ Action Required:** Please find the actual Min/Max values for each feature from your original 
    training dataset and update the `SCALER_PARAMS` dictionary in `app.py` for accurate results.
    """)

    # Create two columns for a better layout
    col1, col2 = st.columns(2)

    # Dictionary to hold user inputs
    user_input_dict = {}

    # Display input fields in columns
    for i, feature in enumerate(FEATURE_COLUMNS):
        # Split the 28 inputs roughly evenly between col1 and col2 (14 inputs per column)
        current_col = col1 if i < len(FEATURE_COLUMNS) / 2 else col2
        
        # Use the full name for the label
        full_name = FEATURE_NAMES.get(feature, feature.replace('_', ' ').title())
        
        params = SCALER_PARAMS.get(feature, {'min': 0.0, 'max': 100.0, 'step': 1.0})
        
        # Use st.slider for percentage/rate features, st.number_input for large counts
        if 'pct' in feature or 'rate' in feature or 'percent' in feature or 'age' in feature:
            input_value = current_col.slider(
                f"**{full_name}**",
                min_value=float(params['min']),
                max_value=float(params['max']),
                value=float(params['min']), # Default to min value
                step=float(params['step']),
                format="%.2f"
            )
        else:
            # Use st.number_input for count/income features
            input_value = current_col.number_input(
                f"**{full_name}**",
                min_value=float(params['min']),
                max_value=float(params['max']),
                value=float(params['min']), # Default to min value
                step=float(params['step']),
                format=f"%.{2 if params['step'] < 1 else 0}f"
            )
            
        user_input_dict[feature] = input_value

    # --- 5. PREDICTION LOGIC ---
    st.markdown("---")
    if st.button('🚀 Predict Average Annual Count', type='primary'):
        if model:
            # 1. Prepare data for scaling
            raw_data = pd.DataFrame([user_input_dict])
            
            # 2. Scale the input data manually using the defined SCALER_PARAMS
            scaled_data = raw_data.copy()
            
            for col in FEATURE_COLUMNS:
                min_val = SCALER_PARAMS[col]['min']
                max_val = SCALER_PARAMS[col]['max']
                
                # Apply MinMax scaling
                scaled_data[col] = raw_data[col].apply(
                    lambda x: min_max_scale_single_value(x, min_val, max_val)
                )

            # 3. Make Prediction
            try:
                # The model expects a single row DataFrame of 28 scaled features
                prediction = model.predict(scaled_data)
                
                # Format and display result
                st.success(f"### Predicted Average Annual Cancer Count: {int(np.round(prediction[0])):,} Cases")
                st.info("Prediction Note: This is an estimated value. Factors like data quality and model fit may affect accuracy.")
                
                # Optional: Display the scaled inputs for debugging
                with st.expander("Show Scaled Inputs"):
                    st.dataframe(scaled_data)
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- NEW: ABOUT PAGE FUNCTION ---
def show_about_page():
    """Displays information about the model and features."""
    st.title("About the Cancer Regression Model")
    st.markdown("""
    This application uses a multiple linear regression model trained on the `cancer_reg.csv` dataset.

    ### Model Details
    * **Algorithm:** Scikit-learn Linear Regression.
    * **Target Variable:** `avganncount` (Average number of cancer cases diagnosed annually).
    * **Input Features:** 28 demographic and public health indicators (listed below).
    * **Preprocessing:** All 28 features were scaled using **MinMaxScaler** before training. The live prediction feature uses the hardcoded min/max values to perform the same scaling on new inputs.
    
    ### Features Used in Prediction
    The model relies on the following 28 features to make a prediction:
    """)
    
    # Use the full name mapping in the About Page as well
    feature_list = [f"* **{FEATURE_NAMES[col]}** (`{col}`)" for col in FEATURE_COLUMNS]
    st.markdown('\n'.join(feature_list))
    
    st.markdown("""
    ---
    ### Deployment Note
    If you did not save the original `MinMaxScaler` object, ensure the `SCALER_PARAMS` dictionary 
    in the application code contains the correct minimum and maximum values extracted 
    from your full training dataset for each feature.
    """)


# --- 4. STREAMLIT UI LAYOUT (Main Execution) ---
st.set_page_config(page_title="Cancer Incidence Prediction", layout="wide")

# Load and apply custom CSS
load_css("styles.css")

# --- NAVIGATION BAR (Sidebar) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "About Model"])
st.sidebar.markdown("---")
st.sidebar.image("https://placehold.co/100x100/4B0082/ffffff?text=ML", caption="ML Model App") # Updated Image color

# --- Page Rendering Logic ---
if page == "Predictor":
    show_predictor_page(model)
elif page == "About Model":
    show_about_page()
