import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.express as px

# ---------------------------------------------------------
# 1. Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva 3D",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard 3D de Manutenção Preditiva")
st.markdown("""
Visualização tridimensional para análise de sensores de chão de fábrica.
Ajuste os parâmetros do **DBSCAN** na barra lateral para recalcular os agrupamentos e identificar anomalias.
""")

# ---------------------------------------------------------
# 2. Geração do Conjunto de Dados Sintético (com Outliers)
# ---------------------------------------------------------
@st.cache_data
def gerar_dados():
    np.random.seed(42)
    n_samples = 300

    # Cluster 1: Funcionamento Padrão / Carga Normal
    vibracao_1 = np.random.normal(loc=50, scale=5, size=n_samples // 2)
    temp_1 = np.random.normal(loc=65, scale=4, size=n_samples // 2)

    # Cluster 2: Funcionamento Alta Carga / Regime Severo
    vibracao_2 = np.random.normal(loc=80, scale=6, size=n_samples // 2)
    temp_2 = np.random.normal(loc=85, scale=5, size=n_samples // 2)

    # Outliers Intencionais (Falhas/Anomalias)
    outliers_vibracao = np.random.uniform(low=20, high=120, size=20)
    outliers_temp = np.random.uniform(low=40, high=110, size=20)

    # Consolidação dos dados
    vibracao = np.concatenate([vibracao_1, vibracao_2, outliers_vibracao])
    temp = np.concatenate([temp_1, temp_2, outliers_temp])
    
    ids_maquinas = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]

    df = pd.DataFrame({
        'ID_Maquina': ids_maquinas,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temp
    })
    
    return df

df = gerar_dados()

# ---------------------------------------------------------
# 3. Barra Lateral (Sidebar) - Controles do DBSCAN
# ---------------------------------------------------------
st.sidebar.header("🔧 Parâmetros do DBSCAN")

eps_val = st.sidebar.slider(
    "Distância Máxima (`eps`):",
    min_value=1.0,
    max_value=20.0,
    value=5.5,
    step=0.5,
    help="Distância máxima entre dois pontos para que sejam considerados vizinhos."
)

min_samples_val = st.sidebar.slider(
    "Mínimo de Amostras (`min_samples`):",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Número mínimo de pontos em uma vizinhança para formar uma região densa."
)

# ---------------------------------------------------------
# 4. Aplicação do DBSCAN
# ---------------------------------------------------------
X = df[['Frequencia_Vibracao', 'Pico_Temperatura']]

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
df['Cluster'] = dbscan.fit_predict(X)

# Status amigável
df['Status'] = df['Cluster'].apply(lambda x: 'Anomalia (Outlier)' if x == -1 else f'Normal (Cluster {x})')

# Métricas resumo
total_anomalias = (df['Cluster'] == -1).sum()
total_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)

col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas", len(df))
col2.metric("Clusters Operacionais", total_clusters)
col3.metric("Anomalias Detectadas", total_anomalias, delta_color="inverse")

# ---------------------------------------------------------
# 5. Gráfico 3D Interativo sem Legenda (Plotly)
# ---------------------------------------------------------
st.subheader("📊 Visualização 3D (Vibração x Temperatura x Cluster)")

# Mapeamento dinâmico de cores (Preto para outliers / Cluster -1)
clusters_unicos = sorted(df['Cluster'].unique())
color_sequence = px.colors.qualitative.Dark24
color_map = {}

idx_cor = 0
for c in clusters_unicos:
    if c == -1:
        color_map[str(c)] = "#000000"  # Preto para Anomalias
    else:
        color_map[str(c)] = color_sequence[idx_cor % len(color_sequence)]
        idx_cor += 1

# Convertemos Cluster para string apenas para o mapeamento de cores do Plotly
df['Cluster_Str'] = df['Cluster'].astype(str)

fig = px.scatter_3d(
    df,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    z='Cluster',
    color='Cluster_Str',
    color_discrete_map=color_map,
    hover_name='ID_Maquina',
    hover_data={
        'Cluster_Str': False,
        'Cluster': True,
        'Frequencia_Vibracao': ':.2f',
        'Pico_Temperatura': ':.2f',
        'Status': True
    },
    labels={
        'Frequencia_Vibracao': 'Frequência Vibração (Hz)',
        'Pico_Temperatura': 'Pico Temperatura (°C)',
        'Cluster': 'Cluster ID'
    }
)

# Configurações de layout: remove a legenda e ajusta marcadores
fig.update_layout(
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title='Frequência (Hz)',
        yaxis_title='Temperatura (°C)',
        zaxis_title='Cluster ID'
    )
)

fig.update_traces(
    marker=dict(size=5, opacity=0.8)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. Tabela Interativa com Filtro de Status
# ---------------------------------------------------------
st.subheader("📋 Tabela de Monitoramento de Máquinas")

# Filtro de exibição
filtro_status = st.radio(
    "Filtrar exibição da tabela:",
    options=["Apenas Anomalias", "Apenas Normais", "Todas as Máquinas"],
    horizontal=True
)

if filtro_status == "Apenas Anomalias":
    df_exibicao = df[df['Cluster'] == -1]
elif filtro_status == "Apenas Normais":
    df_exibicao = df[df['Cluster'] != -1]
else:
    df_exibicao = df

st.dataframe(
    df_exibicao[['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura', 'Status', 'Cluster']].style.format({
        'Frequencia_Vibracao': '{:.2f} Hz',
        'Pico_Temperatura': '{:.2f} °C'
    }),
    use_container_width=True,
    hide_index=True
)
