import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Зареждане на 10-те файла (моделите и скалъра)
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    models = {
        'Logistic Regression': joblib.load('log_reg.pkl'),
        'Ridge': joblib.load('ridge.pkl'),
        'LASSO': joblib.load('lasso.pkl'),
        'Elastic Net': joblib.load('elastic.pkl'),
        'Classification Tree': joblib.load('tree.pkl'),
        'Random Forest': joblib.load('rf.pkl'),
        'XGBoost': joblib.load('xgb.pkl'),
        'Neural Network': joblib.load('nn.pkl')
    }
    return scaler, models

scaler, models = load_models()

st.title("Визуализатор за Риск от csPCa (ISUP > 1)")
st.markdown("Въведете клиничните данни на пациента. Системата автоматично ще изчисли **PSAd lesion** и ще анализира риска чрез 8 AI алгоритъма.")

# 2. Полета за въвеждане от лекаря
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Възраст (Age)", min_value=40, max_value=100, value=65)
    tpsa = st.number_input("tPSA (ng/mL)", min_value=0.1, max_value=100.0, value=5.0, format="%.2f")
    pv = st.number_input("Обем на простатата (PV в mL)", min_value=10, max_value=200, value=50)

with col2:
    pirads = st.selectbox("PI-RADS Скор", options=[1, 2, 3, 4, 5], index=2)
    lesion_vol = st.number_input("Обем на лезията (mL)", min_value=0.1, max_value=50.0, value=1.0, format="%.2f")

# 3. Автоматично изчисляване на PSAd lesion
psad_lesion = tpsa / lesion_vol
st.info(f"**Автоматично изчислен PSAd lesion:** {psad_lesion:.3f}")

# Подготовка на данните СТРИКТНО в реда, в който са тренирани моделите (5 променливи!)
feature_names = ['Age', 'tPSA', 'PV', 'PI-RADS', 'PSAd lesion']
patient_data = np.array([[age, tpsa, pv, pirads, psad_lesion]])
patient_df = pd.DataFrame(patient_data, columns=feature_names)

# Мащабиране (Scaling) за линейните модели и Невронната мрежа
patient_scaled = scaler.transform(patient_df)

# Бутон за пресмятане
if st.button("Изчисли Риска", type="primary"):
    st.divider()
    st.subheader("Резултати от 8-те алгоритъма:")
    
    # Отпечатване на 8-те процента в решетка
    cols = st.columns(4)
    model_keys = list(models.keys())
    
    for i, m_name in enumerate(model_keys):
        model = models[m_name]
        
        # Линейните и NN ползват мащабирани данни, дърветата ползват оригинални
        if m_name in ['Logistic Regression', 'Ridge', 'LASSO', 'Elastic Net', 'Neural Network']:
            prob = model.predict_proba(patient_scaled)[0][1] * 100
        else:
            prob = model.predict_proba(patient_df)[0][1] * 100
            
        with cols[i % 4]:
            st.metric(label=m_name, value=f"{prob:.1f}%")
            
    st.divider()
    
    # 4. Визуализация на влиянието (SHAP Waterfall) за Random Forest
    st.subheader("Обяснение на решението (Random Forest)")
    st.markdown("Графиката показва как всеки индивидуален параметър е повлиял за повишаване (червено) или понижаване (синьо) на риска за този конкретен пациент спрямо средния риск.")
    
    rf_model = models['Random Forest']
    
    # Генериране на SHAP стойности
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(patient_df)
    
    # Извличане на стойностите за клас 1 (csPCa)
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1][0]
        base_val = explainer.expected_value[1]
    else:
        sv_class1 = shap_values[0]
        base_val = explainer.expected_value
        
    # Построяване на графиката
    explanation = shap.Explanation(values=sv_class1, 
                                   base_values=base_val, 
                                   data=patient_df.iloc[0].values, 
                                   feature_names=feature_names)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)