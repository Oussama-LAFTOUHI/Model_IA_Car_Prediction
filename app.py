import streamlit as st
import joblib, pandas as pd, numpy as np

model = joblib.load('modele_voiture.pkl')
colonnes = list(model.feature_names_in_)
marques = [c.replace('Marque_' , '') for c in colonnes if 'Marque_' in c]
models = [c.replace('Modele_','') for c in colonnes if 'Modele_' in c ]
st.title("prediction de votre voiture")
km = st.number_input("Km",0,500000,20000)
anne = st.number_input("Annee",1990,2026,2021)
marque = st.selectbox("Marque" , marques)
Model = st.selectbox("Model", models)
if st.button("Start"):
    df=pd.DataFrame([[km , anne , marque , Model]] , columns=['Km' , 'Annee' , 'Marque', 'Modele'])
    df=pd.get_dummies(df)
    df=df.reindex(columns=colonnes , fill_value=0)
    prix = np.exp(model.predict(df)[0])
    st.success(f"Prix {prix:,.0f} DH")
