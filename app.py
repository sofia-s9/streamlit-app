import pickle
import os
import streamlit as st
import pandas as pd
import numpy as np

# Настройка страницы 
st.set_page_config(
    page_title="Анализ данных: Принцессы Диснея",
    layout="centered"
)

WEIGHTS_PATH = os.path.join("data", "model_weights.mw")
CSV_PATH = "disney_princess_popularity_dataset_300_rows.csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def load_artifacts():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    with open(WEIGHTS_PATH, "rb") as f:
        return pickle.load(f)

df = load_data()
artifacts = load_artifacts()

# Заголовок
st.title("Исследование популярности Принцесс Диснея")

tab1, tab2, tab3 = st.tabs(["Описательный анализ", "Проверка моделей", "Экспресс-прогноз"])

with tab1:
    st.header("1. Первичная описательная статистика")
    
    # Краткие метрики
    m1, m2, m3 = st.columns(3)
    m1.metric("Объем выборки (N)", f"{df.shape[0]} строк")
    m2.metric("Кол-во признаков (P)", df.shape[1])
    m3.metric("Уникальных имен", df["PrincessName"].nunique())
    
    with st.expander("Просмотр сырых данных (первые 5 строк)"):
        st.dataframe(df.head(5), width="stretch")
        
    st.subheader("Распределение целевой переменной (PopularityScore)")
    counts, bins = np.histogram(df["PopularityScore"], bins=15)
    hist_df = pd.DataFrame({"Количество": counts}, index=np.round(bins[:-1], 1))
    st.bar_chart(hist_df, color="#4ca3a3")
    
    st.subheader("Связь признаков: Сборы vs Популярность")
    st.caption("Интерактивная диаграмма рассеяния. Попробуйте выделить область мышкой.")
    st.scatter_chart(
        df, 
        x="BoxOfficeMillions", 
        y="PopularityScore", 
        color="IsIconic",
        size="IMDB_Rating",
        width="stretch"
    )
    
    st.subheader("Временной тренд по эпохам")
    trend = df.groupby("FirstMovieYear")["PopularityScore"].mean().reset_index()
    st.line_chart(trend.set_index("FirstMovieYear"), color="#f28e2b")

with tab2:
    st.header("2. Оценка качества математических моделей")
    
    if artifacts is None:
        st.warning("⚠️ Не найдены обученные модели. Сначала запустите `python model.py` в консоли.")
        st.stop()
        
    m = artifacts["metrics"]
    
    st.subheader("Метрики задачи регрессии (Ridge Regression)")
    r_col1, r_col2 = st.columns(2)
    r_col1.metric("R² на кросс-валидации (CV-5)", f"{m['reg']['cv']:.4f}")
    r_col2.metric("Baseline R² (прогноз по среднему)", f"{m['reg']['baseline']:.4f}")
    
    st.subheader("Метрики задачи классификации (Random Forest)")
    c_col1, c_col2 = st.columns(2)
    c_col1.metric("Accuracy на тесте", f"{m['clf']['test']:.4f}")
    c_col2.metric("Базовая Accuracy (если заполнять модами)", f"{m['clf']['baseline']:.4f}")
    
    st.info("💡 **Вывод:** Метрики качества моделей крайне близки к константным базовым-прогнозам (прогнозу по среднему значению). Коэффициент детерминации R^2 примерно равный 0 указывает на отсутствие сильной линейной или нелинейной связи между сгенерированными признаками и таргетом. Это явный признак того, что предоставленный датасет имеет **синтетическую (зашумленную) природу**, где признаки генерировались независимо от целевых переменных.")

with tab3:
    st.header("3. Симуляция прогноза для новой принцессы")
    st.markdown("Выделили 6 ключевых метрик. Остальные параметры заполняются автоматически средними значениями выборки.")
    
    if artifacts is None:
        st.stop()
        
    with st.form("short_predict_form"):
        col_f1, col_f2 = st.columns(2)
        
        box_office = col_f1.number_input("Кассовые сборы фильма (млн $)", min_value=0, max_value=2000, value=500)
        imdb_rating = col_f1.slider("Рейтинг IMDB", 1.0, 10.0, 7.2, step=0.1)
        tiktok_views = col_f1.number_input("Просмотры в TikTok (млн)", min_value=0, value=300)
        
        google_idx = col_f2.slider("Google Search Index", 0, 100, 50)
        screen_time = col_f2.slider("Экранное время принцессы (мин)", 10, 120, 45)
        num_songs = col_f2.selectbox("Количество песен", [1, 2, 3, 4, 5, 6, 7], index=2)
        
        submit = st.form_submit_button("Рассчитать прогноз моделей", width="stretch")
        
    if submit:
        input_row = df.drop(columns=["PopularityScore", "IsIconic"], errors="ignore").iloc[[0]].copy()
        
        for col in input_row.columns:
            if df[col].dtype == "object" or df[col].dtype == "string":
                input_row[col] = df[col].mode()[0]
            else:
                input_row[col] = df[col].median()
                
        # Внедряем то, что ввёл пользователь
        input_row["BoxOfficeMillions"] = box_office
        input_row["IMDB_Rating"] = imdb_rating
        input_row["TikTokHashtagViewsMillions"] = tiktok_views
        input_row["GoogleSearchIndex2024"] = google_idx
        input_row["AvgScreenTimeMinutes"] = screen_time
        input_row["NumberOfSongs"] = num_songs
        
        # Делаем предсказания
        reg_pred = artifacts["reg_model"].predict(input_row)[0]
        clf_pred = artifacts["clf_model"].predict(input_row)[0]
        clf_proba = artifacts["clf_model"].predict_proba(input_row)[0][1] # вероятность "Yes" (обычно второй класс)
        
        st.markdown("---")
        st.subheader("Результаты симуляции")
        
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Предсказанный PopularityScore", f"{reg_pred:.2f}")
        res_c2.metric("Статус IsIconic?", f"{clf_pred}", delta=f"P(Yes) = {clf_proba:.2f}")
        
        st.warning(f"**Напоминание:** Из-за низкой предсказательной силы модели (см. Вкладку 2), данный прогноз стремится к математическому ожиданию выборки ({artifacts['data_summary']['mean_pop']:.1f}) и не может быть улучшен без добавления реальных признаков.")
