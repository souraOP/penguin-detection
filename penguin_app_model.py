import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st

st.write("""
# Penguin Prediction App

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
                    

""")

# collect user input features into a dataframe

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type =["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ("Dream", "Biscoe", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("male", "female"))
        bill_length_mm = st.sidebar.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body Mass (g)", 2700.0, 6300.0, 4207.0)
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# combining user input features with the penguin datasets

penguins_data = pd.read_csv("datasets/penguin_data/penguins_cleaned.csv")
penguins = penguins_data.drop(columns=['species'])

df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']

for i in encode:
    dummy = pd.get_dummies(df[i], prefix=i)
    df = pd.concat([df, dummy], axis=1)
    del df[i]

df = df[:1]

st.subheader("User Input")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently using example input features (shown below).")
    st.write(df)

load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# make predictions

prediction = load_clf.predict(df)

pred_probability = load_clf.predict_proba(df)

st.subheader("Predictions")

penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])

st.write(penguin_species[prediction])

st.subheader("Prediction Probability")

st.write(pred_probability)