import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Marks Predictor",page_icon="📝")

st.title("Student Marks Predictor")
st.write("Enter The Number Of Hours Studied (1-10) And **Click Predict** To See The Predicted Marks.")

def load_model(path:str="model.pkl"):
    with open(path,"rb")as f:
        model = pickle.load(f)
    return model

# Load The Model
try:
    model = load_model("model.pkl")
except FileNotFoundError:
    st.error("my_data.pkl not found. Please place your pickle named 'my_data.pkl' in the same folder as app.py")
    st.stop()
except Exception as e:
    st.error(f"Failed To Load Model : {e}")
    st.stop()

# Input : integer / float hours from 1 to 10

hours = st.number_input(
    label = "Hours_Studied",
    min_value = 1.0,
    max_value = 10.0,
    step = 0.1,
    value = 1.0,
    format ="%.1f"
)

# Predict Button

if st.button("Predict"):
    try:
        X = np.array([[hours]],dtype=float)
        prediction = model.predict(X)
        predicted_marks = prediction[0]
 
        # show result
        st.success(f"Predicted Marks : {predicted_marks:.1f}")
        st.write("Note: This is model's Prediction")
    except Exception as e:
        st.error(f"Prediction Failed: {e}")