import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

#load trained model
model = tf.keras.models.load_model('model.h5')


# Encoder and scaler

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

#user input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_ )
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
cred_score = st.number_input('Credit Score')
est_sal = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
no_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#prepare input data
input_data = pd.DataFrame(
    {
    'CreditScore':[cred_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[no_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[est_sal]
})

#onehot encode
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the features
input_data_scaled = scaler.transform(input_data)


#make prediction
prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

#display result
st.write(f'Churn Probablity: {pred_prob:.2f}')

if pred_prob > 0.5:
    st.write('🚨 The customer is likely to churn.')
else:
    st.write('✅ The customer is not likely to churn.')