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

# РЕЧНИК С ПРАГОВЕ ЗА БЕЗОПАСНОСТ (5% изпуснати карциноми)
CUTOFFS = {
    'Logistic Regression': 26.17,
    'Ridge': 26.17,
    'LASSO': 26.17,
    'Elastic Net': 26.41,
    'Classification Tree': 27.78,
    'Random Forest': 32.18,
    'XGBoost': 28.46,
    'Neural Network': 19.25
}

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
    
    # Речник за предварително запазване на изчислените вероятности за всички модели
    calc_probs = {}
    for m_name in models.keys():
        model = models[m_name]
        if m_name in ['Logistic Regression', 'Ridge', 'LASSO', 'Elastic Net']:
            calc_probs[m_name] = model.predict_proba(patient_linear_scaled)[0][1] * 100
        elif m_name == 'Neural Network':
            calc_probs[m_name] = model.predict_proba(patient_tree_scaled)[0][1] * 100
        else: # Дървета
            patient_tree_df_final = pd.DataFrame(patient_tree_imp, columns=feature_names_tree)
            calc_probs[m_name] = model.predict_proba(patient_tree_df_final)[0][1] * 100

    # СПЕЦИАЛНА ФУНКЦИЯ ЗА ОЦВЕТЯВАНЕ В ЧЕРВЕНО
    def display_metric_with_threshold(name, display_name, prob, cutoff):
        if prob >= cutoff:
            # Ако рискът е над прага - червен и удебелен текст
            st.markdown(f"""
            <div style='line-height: 1.2; margin-bottom: 8px;'>
                <span style='font-size: 14px; color: #ff2b2b; font-weight: bold;'>{display_name}</span><br>
                <span style='font-size: 1.8rem; color: #ff2b2b; font-weight: bold;'>{prob:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Ако рискът е под прага - нормален дизайн
            st.metric(label=display_name, value=f"{prob:.1f}%")
        
        # Легенда за прага под самия резултат
        st.caption(f"Праг за изчакване: < {cutoff:.2f}%")

    # ==========================================
    # ГРУПА 1: Конвенционална биостатистика
    # ==========================================
    st.subheader("1. Конвенционална биостатистика")
    st.markdown("Класическият златен стандарт, показващ линейна зависимост между клиничните параметри и риска.")
    
    cols_group1 = st.columns(2) 
    with cols_group1[0]:
        display_metric_with_threshold('Logistic Regression', 'Multivariate Logistic Regression', calc_probs['Logistic Regression'], CUTOFFS['Logistic Regression'])
            
    st.write("") 
    
    # ==========================================
    # ГРУПА 2: Регуляризирани регресии
    # ==========================================
    st.subheader("2. Алгоритми за машинно надзиравано обучение")
    st.markdown("Модели, които съчетават статистика с техники от машинното обучение. Прилагат регуляризация върху регресионните модели, за да се предотврати претрениране когато има много променливи и сравнително малко пациенти.")
    
    cols_group2 = st.columns(3) 
    reg_models = ['Ridge', 'LASSO', 'Elastic Net']
    
    for i, m_name in enumerate(reg_models):
        with cols_group2[i]:
            display_metric_with_threshold(m_name, m_name, calc_probs[m_name], CUTOFFS[m_name])
            
    st.write("") 

    # ==========================================
    # ГРУПА 3: Изкуствен интелект (Машинно обучение)
    # ==========================================
    st.subheader("3. Изкуствен интелект (AI алгоритми)")
    st.markdown("AI алгоритми, способни да откриват сложни скрити нелинейни зависимости, които убягват на конвенционалната статистика.")
    
    cols_group3 = st.columns(4) 
    ai_models = ['Classification Tree', 'Random Forest', 'XGBoost', 'Neural Network']
    
    for i, m_name in enumerate(ai_models):
        with cols_group3[i]:
            display_metric_with_threshold(m_name, m_name, calc_probs[m_name], CUTOFFS[m_name])
            
    st.divider()

    # ==========================================
    # КЛИНИЧЕН КОНСИЛИУМ (СВЕТОФАР)
    # ==========================================
    st.subheader("Клиничен консилиум (Ensemble Voting)")
    st.caption("Обяснение на цветовия код:\n"
               "🟢 Зелено: 0 модела над прага (Пълен консенсус за безопасност).\n"
               "🟡 Жълто: 1 до 4 модела над прага (Дискордантност - изисква се индивидуална експертна преценка).\n"
               "🔴 Червено: 5 до 8 модела над прага (Мнозинството алгоритми алармират за биопсия).")
    
    # Преброяване колко модела светят в червено
    models_over_threshold = 0
    for m in models.keys():
        if calc_probs[m] >= CUTOFFS[m]:
            models_over_threshold += 1
            
    if models_over_threshold >= 5:
        st.error(f"🔴 **СИГНАЛ ЗА БИОПСИЯ (Консенсус: {models_over_threshold} от 8 модела)**\n\nМнозинството алгоритми класифицират пациента **над безопасния праг**. Строго се препоръчва извършване на биопсия, за да се предотврати пропускане на клинично значим карцином.")
    elif models_over_threshold >= 1:
        st.warning(f"🟡 **ГРАНИЧЕН СЛУЧАЙ (Консенсус: {models_over_threshold} от 8 модела)**\n\nЛипсва пълен консенсус между алгоритмите. Тъй като част от моделите показват повишен риск, решението за биопсия трябва да се вземе на базата на индивидуална експертна преценка.")
    else:
        st.success(f"🟢 **БЕЗОПАСНО ИЗЧАКВАНЕ (Консенсус: 0 от 8 модела)**\n\nВсички 8 модела класифицират пациента **под прага за безопасност**. Според данните от проучването, биопсията при този пациент може да бъде безопасно спестена (допустим риск < 5%).")

    st.divider()
    
    # ==========================================
    # 4. Визуализация на влиянието (SHAP Waterfall)
    # ==========================================
    st.subheader("Обяснение на AI решението (SHAP Waterfall)")
    st.markdown("Графиката показва как всяка стойност на пациента е повлияла за повишаване (червено) или понижаване (синьо) на риска.")
    
    # Падащо меню (Само за дърветата засега)
    shap_model_choice = st.selectbox(
        "Изберете модел за визуализация:",
        ['Random Forest', 'XGBoost', 'Classification Tree'],
        index=0
    )
    
    selected_model = models[shap_model_choice]
    patient_tree_df_final = pd.DataFrame(patient_tree_imp, columns=feature_names_tree)
    
    # Изчисляване на SHAP
    explainer = shap.TreeExplainer(selected_model)
    shap_obj = explainer(patient_tree_df_final)
    
    if len(shap_obj.values.shape) == 3:
        exp_single = shap_obj[0, :, 1]
    else:
        exp_single = shap_obj[0]
        
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(exp_single, show=False)
    st.pyplot(fig)
