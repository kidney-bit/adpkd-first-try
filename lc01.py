import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

# Use um caminho relativo para o arquivo CSV
file_path = "lc01_corrigido.csv"

# Carregando os dados corrigidos a partir do caminho relativo
df_reviews = pd.read_csv(file_path)

# Convertendo 'segunda_osm' para numérico, tratando valores inválidos como NaN
df_reviews['segunda_osm'] = pd.to_numeric(df_reviews['segunda_osm'], errors='coerce')

# Remover colunas sensíveis
df_reviews = df_reviews.drop(columns=['RGHC', 'Paciente', 'Data'], errors='ignore')

# Identificando as variáveis numéricas e categóricas
numerical_columns = df_reviews.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = ['mayo_classification', 'diabetes']  # Manter apenas as variáveis categóricas necessárias

# Definindo a ordem fixa para Mayo Classification
mayo_order = ['1A', '1B', '1C', '1D', '1E']

# Sidebar
st.sidebar.header("Filtros")
st.sidebar.write("Use os sliders e checkboxes para filtrar os dados.")

# Sliders para variáveis numéricas contínuas com checkbox (desativados por padrão)
slider_values = {}
apply_filters = {}
for col in numerical_columns:
    apply_filters[col] = st.sidebar.checkbox(f"Ativar filtro para {col}", value=False)
    if apply_filters[col]:
        min_value = df_reviews[col].min()
        max_value = df_reviews[col].max()
        slider_values[col] = st.sidebar.slider(f"{col}", float(min_value), float(max_value), (float(min_value), float(max_value)))

# Checkboxes para variáveis categóricas
selected_categories = {}
for col in categorical_columns:
    unique_values = df_reviews[col].dropna().unique()
    selected_categories[col] = []
    st.sidebar.write(f"**{col}**")
    for val in unique_values:
        if st.sidebar.checkbox(f"{val}", value=True):
            selected_categories[col].append(val)

# Aplicando os filtros numéricos e categóricos
df_filtered = df_reviews.copy()

# Filtrando variáveis numéricas apenas se o filtro estiver ativado e omitindo valores ausentes
for col, is_active in apply_filters.items():
    if is_active and col in slider_values:
        min_val, max_val = slider_values[col]
        df_filtered = df_filtered[(df_filtered[col] >= min_val) & (df_filtered[col] <= max_val)]

# Filtrando variáveis categóricas
for col, selected_vals in selected_categories.items():
    if selected_vals:
        df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]

# Contador Dinâmico de Pacientes
num_pacientes = df_filtered.shape[0]
st.markdown(f'<p class="font-32">Número de Pacientes Filtrados: {num_pacientes}</p>', unsafe_allow_html=True)

# ======================== Exibindo a Tabela Filtrada ========================
st.markdown('<p class="font-32">Tabela Filtrada</p>', unsafe_allow_html=True)
st.write(df_filtered)

# ======================== Gráfico Dinâmico de Contagem por Mayo Class ========================
st.markdown('<p class="font-32">Contagem de Pacientes por Classificação Mayo</p>', unsafe_allow_html=True)

# Gráfico de contagem dinâmica com ordem fixa
if 'mayo_classification' in df_filtered.columns:
    mayo_counts = df_filtered['mayo_classification'].value_counts().reindex(mayo_order, fill_value=0).reset_index()
    mayo_counts.columns = ['mayo_classification', 'count']
    fig_mayo_counts = px.bar(mayo_counts, x='mayo_classification', y='count',
                             labels={'mayo_classification': 'Classificação Mayo', 'count': 'Contagem'},
                             title="Contagem de Pacientes por Classificação Mayo (Filtrada)",
                             category_orders={"mayo_classification": mayo_order})
    st.plotly_chart(fig_mayo_counts)

# ======================== Análise de Regressão Multivariada 1 ========================
st.markdown('<p class="font-32">Análise de Regressão Multivariada: Previsão de CKD-EPI 2021</p>', unsafe_allow_html=True)

# Convertendo as variáveis categóricas para códigos numéricos, se necessário
df_filtered['mayo_classification'] = pd.Categorical(df_filtered['mayo_classification'], categories=mayo_order).codes

X = df_filtered[['idade', 'mayo_classification']]
y = df_filtered['ult_ckd_epi2021']

model1 = LinearRegression()
model1.fit(X, y)

# Prevendo os valores de CKD-EPI 2021 com o modelo ajustado
df_filtered['predicted_ckd_epi'] = model1.predict(X)

# Exibindo os coeficientes da regressão
st.markdown(f'<p class="font-26">Intercepto: {model1.intercept_}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="font-26">Coeficientes: {model1.coef_}</p>', unsafe_allow_html=True)

# Criando um gráfico de linhas para visualizar as previsões da regressão
df_filtered['mayo_classification_str'] = df_filtered['mayo_classification'].astype(str)

fig_line_ckd = px.line(df_filtered.sort_values(by='idade'), x='idade', y='predicted_ckd_epi', color='mayo_classification_str',
                       labels={'idade': 'Idade', 'predicted_ckd_epi': 'CKD-EPI 2021 Prevendo', 'mayo_classification_str': 'Classificação Mayo'},
                       title="Previsão de CKD-EPI 2021 com Base na Idade e Classificação Mayo (Regressão Multivariada)")

# Exibindo o gráfico de linhas
st.plotly_chart(fig_line_ckd)
