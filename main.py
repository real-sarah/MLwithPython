import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and the scaler
pipe = pickle.load(open("finalized_model.pickle", 'rb'))
scaler = pickle.load(open("scaler_model.pickle", "rb"))

# Define the main function for the Streamlit app
def main():
    st.title("University Admission Prediction")

    # Display an image in the center
    st.image("https://tse1.mm.bing.net/th?id=OIP.h2gG87iB1-mfwYJfMMEepQHaCD&pid=Api&P=0&h=180")


    # Collect user inputs using interactive widgets
    gres = st.number_input("***Enter your GRE score (out of 340):***", min_value=0, max_value=340, step=1, format="%d")
    toefl = st.number_input("***Enter your TOEFL score (out of 120):***", min_value=0, max_value=120, step=1, format="%d")
    rating = st.number_input("***Enter your University Rating (out of 5):***", min_value=0.0, max_value=5.0, step=0.1,
                             format="%.1f")
    sop = st.number_input("***Enter your SOP strength (out of 5):***", min_value=0.0, max_value=5.0, step=0.1, format="%.1f")
    lor = st.number_input("***Enter your LOR strength (out of 5):***", min_value=0.0, max_value=5.0, step=0.1, format="%.1f")
    cgpa = st.number_input("***Enter your CGPA (out of 10):***", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")

    # Toggle for research experience
    research = st.selectbox("***Do you have research experience?***", ("Yes", "No"))
    research = 1 if research == "Yes" else 0

    # Prediction button
    if st.button("***Predict Admission Chances***"):
        # Check for input validation
        if gres == "" or toefl == "" or rating == "" or sop == "" or lor == "" or cgpa == "":
            st.warning("Please fill in all the fields before submitting!")
        elif cgpa > 10 or gres > 340 or toefl > 120 or sop > 5 or lor > 5:
            st.warning("Invalid inputs: Please ensure all values are within valid ranges.")
        else:
            try:
                # Prepare the input DataFrame for prediction
                input_data = pd.DataFrame([[gres, toefl, rating, sop, lor, cgpa, research]],
                                          columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])

                # Standardize the input data using the scaler
                standardized_data = scaler.transform(input_data)

                # Predict using the loaded model
                prediction = pipe.predict(standardized_data)[0]

                # Convert the prediction to a percentage and ensure it's within 0 to 100%
                prediction_percentage = min(max(prediction * 100, 0), 100)

                # Display the result
                st.success(f"Your predicted chance of admission is {round(prediction_percentage, 2)}%.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Entry point to run the Streamlit app
if __name__ == '__main__':
    main()
