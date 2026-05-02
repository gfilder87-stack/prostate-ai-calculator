import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Зареждане на файловете (Модели, скалъри и списъци с колони)
@st.cache_resource
def load_models():
    imputer_tree = joblib.load('imputer_tree.pkl')
    scaler_tree = joblib.load('scaler_tree.pkl')
    scaler_linear = joblib.load('scaler_linear.pkl')
    linear_cols = joblib.load('linear_feature_names.pkl')
    
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
    return imputer_tree, scaler_tree, scaler_linear, linear_cols, models

imputer_tree, scaler_tree, scaler_linear, linear_cols, models = load_models()

# Центрирани заглавие и академично описание
st.markdown("<h1 style='text-align: center;'>Калкулатор за риск от клинично значим карцином на простатната жлеза (ISUP ≥ 2)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Въведете клиничните данни на пациента. Системата автоматично ще изчисли плътността на tPSA в лезията и ще стратифицира риска според 8 алгоритъма.</p>", unsafe_allow_html=True)
st.divider()

# 2. Полета за въвеждане от лекаря
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Възраст (години)", min_value=40, max_value=100, value=65)
    tpsa = st.number_input("tPSA (ng/mL)", min_value=0.1, max_value=100.0, value=5.0, format="%.2f")
    pv = st.number_input("Обем на простатата (mm³)", min_value=10, max_value=200, value=50)

with col2:
    pirads = st.selectbox("PI-RADS оценка", options=[2, 3, 4, 5], index=1)
    lesion_vol = st.number_input("Обем на лезията (mm³)", min_value=0.1, max_value=50.0, value=1.0, format="%.2f")
    
    # 3. Автоматично изчисляване (Позиционирано в колона 2, ред 3)
    psad_lesion = (tpsa - (0.12 * pv)) / lesion_vol
    st.info(f"**Автоматично изчислена плътност на tPSA в лезията:** {psad_lesion:.2f}")

# --- ПОТОК 1: Подготовка на данни за Дървета и Невронна мрежа ---
feature_names_tree = ['Age', 'tPSA', 'PV', 'PI-RADS', 'PSAd lesion']
patient_df_tree = pd.DataFrame([[age, tpsa, pv, pirads, psad_lesion]], columns=feature_names_tree)

patient_tree_imp = imputer_tree.transform(patient_df_tree)
patient_tree_scaled = scaler_tree.transform(patient_tree_imp)

# --- ПОТОК 2: Подготовка на данни за Линейните модели (Логаритми и Dummy) ---
patient_linear_df = pd.DataFrame(0, index=[0], columns=linear_cols)
patient_linear_df['Age'] = age
patient_linear_df['log_tPSA'] = np.log(tpsa)
patient_linear_df['log_PV'] = np.log(pv)
patient_linear_df['PSAd lesion'] = psad_lesion

pirads_col = f'PIRADS_{pirads}'
if pirads_col in linear_cols:
    patient_linear_df[pirads_col] = 1

patient_linear_scaled = scaler_linear.transform(patient_linear_df)

# Бутон за пресмятане
if st.button("Изчисли риска", type="primary", use_container_width=True):
    st.divider()
    
    # ==========================================
    # ГРУПА 1: Конвенционална биостатистика
    # ==========================================
    st.subheader("1. Конвенционална биостатистика")
    st.markdown("Мултивариантна логистична регресия. Класическият златен стандарт, показващ базовата линейна зависимост между клиничните параметри и риска.")
    
    cols_group1 = st.columns(2) 
    model_log = models['Logistic Regression']
    prob_log = model_log.predict_proba(patient_linear_scaled)[0][1] * 100
    with cols_group1[0]:
        st.metric(label="Multivariate Logistic Regression", value=f"{prob_log:.1f}%")
            
    st.write("") 
    
    # ==========================================
    # ГРУПА 2: Регуляризирани регресии
    # ==========================================
    st.subheader("2. Регуляризирани регресии")
    st.markdown("Модели, които съчетават статистика с техники от машинното обучение. Те прилагат регуляризация върху регресионните модели, за да се предотврати претрениране (overfitting).")
    
    cols_group2 = st.columns(3) 
    reg_models = ['Ridge', 'LASSO', 'Elastic Net']
    
    for i, m_name in enumerate(reg_models):
        model = models[m_name]
        prob = model.predict_proba(patient_linear_scaled)[0][1] * 100
        with cols_group2[i]:
            st.metric(label=m_name, value=f"{prob:.1f}%")
            
    st.write("") 

    # ==========================================
    # ГРУПА 3: Изкуствен интелект (Машинно обучение)
    # ==========================================
    st.subheader("3. Изкуствен интелект (AI алгоритми)")
    st.markdown("Модерни нелинейни AI алгоритми, способни да откриват сложни скрити зависимости, които убягват на конвенционалната статистика.")
    
    cols_group3 = st.columns(4) 
    ai_models = ['Classification Tree', 'Random Forest', 'XGBoost', 'Neural Network']
    
    rf_prob_value = 0 # Променлива, в която ще запазим резултата на Random Forest
    
    for i, m_name in enumerate(ai_models):
        model = models[m_name]
        
        if m_name == 'Neural Network':
            prob = model.predict_proba(patient_tree_scaled)[0][1] * 100
        else: # Дървета
            patient_tree_df_final = pd.DataFrame(patient_tree_imp, columns=feature_names_tree)
            prob = model.predict_proba(patient_tree_df_final)[0][1] * 100
            
        # Записваме специфично резултата на Random Forest за светофара
        if m_name == 'Random Forest':
            rf_prob_value = prob
            
        with cols_group3[i]:
            st.metric(label=m_name, value=f"{prob:.1f}%")
            
    st.divider()

    # ==========================================
    # КЛИНИЧНА ПРЕПОРЪКА (СВЕТОФАР)
    # ==========================================
    st.subheader("Клинична препоръка (базирана на модела-шампион Random Forest)")
    
    rf_cutoff = 32.18 # Доказаният праг от Вашата статия
    
    if rf_prob_value >= rf_cutoff:
        st.error(f"🔴 **СИГНАЛ ЗА БИОПСИЯ (Индивидуален риск: {rf_prob_value:.1f}%)**\n\nРискът на пациента е **над прага за безопасност от {rf_cutoff}%**. Препоръчва се извършване на биопсия, за да се гарантира под 5% риск от пропускане на клинично значим карцином.")
    else:
        st.success(f"🟢 **БЕЗОПАСНО ИЗЧАКВАНЕ (Индивидуален риск: {rf_prob_value:.1f}%)**\n\nРискът на пациента е **под прага за безопасност от {rf_cutoff}%**. Според валидационните данни от проучването, биопсията при този пациент може да бъде безопасно спестена (допустим риск от изпускане на csPCa < 5%).")

    st.divider()
    
    # ==========================================
    # 4. Визуализация на влиянието (SHAP Waterfall)
    # ==========================================
    st.subheader("Обяснение на AI решението (Random Forest SHAP)")
    st.markdown("Графиката показва как всяка стойност на различните параметри е повлияла за повишаване (червено) или понижаване (синьо) на индивидуалния риск спрямо средния базов риск в кохортата.")
    
    rf_model = models['Random Forest']
    patient_tree_df_final = pd.DataFrame(patient_tree_imp, columns=feature_names_tree)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_obj = explainer(patient_tree_df_final)
    
    if len(shap_obj.values.shape) == 3:
        exp_single = shap_obj[0, :, 1]
    else:
        exp_single = shap_obj[0]
        
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(exp_single, show=False)
    st.pyplot(fig)
