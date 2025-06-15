import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import plotly.graph_objects as go

model_rfr = joblib.load("train/models/rfr.pkl")

with open("train/models/xgb.pkl", "rb") as f:
    model_xgb = pickle.load(f)

def prediksi_harga_emas_lengkap(model, input_fitur, harga_emas_hari_ini):
    fitur = ['SPX', 'USO', 'SLV', 'EUR/USD', 'GLD', 'GLD_t-1']
    input_df = pd.DataFrame([input_fitur], columns=fitur)
    pred = model.predict(input_df)[0]
    selisih = pred - harga_emas_hari_ini
    status = "Naik" if selisih > 0 else "Turun" if selisih < 0 else "Tetap"
    return pred, status, abs(selisih)

st.set_page_config(layout="wide")

st.title("Prediksi Harga Emas")

col1, col2 = st.columns(2)

with col1:
    spx = st.number_input("Indeks saham")
    uso = st.number_input("Indikator harga minyak mentah")
    slv = st.number_input("Indikator harga perak")
    eur_usd = st.number_input("EUR/USD: Nilai Tukar Euro terhadap Dolar AS")
    gld = st.number_input("Indikator harga emas hari ini")
    gld_t1 = st.number_input("Indikator harga emas kemarin")
    predict_button = st.button("Predict")

with col2:
    if predict_button:
        st.subheader("Hasil Metode Random Forest Regression :chart_with_upwards_trend:", divider=True)
        rows = st.container(border=True)
        features = np.array([[spx,uso,slv,eur_usd, gld, gld_t1]])
        prediction_rfr = model_rfr.predict(features)
        rows.write(f"Prediksi harga emas besok dengan Random Forest Regression: **{prediction_rfr[0]:.2f} USD**")
        _, status, selisih = prediksi_harga_emas_lengkap(model_rfr, features[0], gld)
        rows.write(f"**Status**: {status} sebesar **{selisih:.2f} USD** dibanding harga saat ini **{gld:.2f} USD**\n")

        st.subheader("Hasil XGBoost :chart_with_upwards_trend:", divider="red")
        prediction_xgb = model_xgb.predict(features)
        _, status, selisih = prediksi_harga_emas_lengkap(model_xgb, features[0], gld)
        rows.write(f"Prediksi harga emas besok dengan XGBoost: **{prediction_xgb[0]:.2f} USD**")
        rows.write(f"**Status**: {status} sebesar **{selisih:.2f} USD** dibanding harga saat ini **{gld:.2f} USD**\n")

        st.subheader("Visualisasi :bar_chart:", divider="green")
        fig = go.Figure(data=[
            go.Bar(name='Harga Hari Ini', x=['Harga'], y=[harga_emas_hari_ini], marker_color='gray'),
            go.Bar(name='Random Forest', x=['Harga'], y=[pred_rf], marker_color='forestgreen'),
            go.Bar(name='XGBoost', x=['Harga'], y=[pred_xgb], marker_color='orange'),
        ])

        fig.update_layout(
            title="Perbandingan Harga Emas Hari Ini dan Prediksi (+1 Hari)",
            yaxis_title="Harga (USD)",
            barmode='group'
        )

        st.plotly_chart(fig)
