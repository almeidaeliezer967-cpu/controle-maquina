import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------
# 1. Configuração da Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva: Detecção de Anomalias")
st.markdown("""
Este aplicativo utiliza o algoritmo **DBSCAN** para identificar comportamentos anômalos em máquinas de chão de fábrica 
com base em leituras de **Frequência de Vibração** e **Pico de Temperatura**.
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
st.sidebar.markdown("Ajuste os hiperparâmetros para otimizar o agrupamento e a detecção de anomalias.")

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

# No DBSCAN, o rótulo -1 representa ruídos/outliers
df['Status'] = df['Cluster'].apply(lambda x: 'Anomalia (Outlier)' if x == -1 else f'Cluster {x}')

# Métrica resumo
total_anomalias = (df['Cluster'] == -1).sum()
total_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)

col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas Monitoradas", len(df))
col2.metric("Clusters Identificados", total_clusters)
col3.metric("Anomalias Detectadas", total_anomalias, delta_color="inverse")

# ---------------------------------------------------------
# 5. Gráfico de Dispersão com Seaborn
# ---------------------------------------------------------
st.subheader("📊 Mapeamento de Clusters e Anomalias")

fig, ax = plt.subplots(figsize=(10, 6))

# Mapeamento de cores: cinza/tons variados para clusters, preto fixo para anomalias (-1)
clusters_unicos = sorted(df['Cluster'].unique())
palette = {}

# Define cores para cada cluster existente
colors = sns.color_palette("tab10", len(clusters_unicos))
color_idx = 0

for c in clusters_unicos:
    if c == -1:
        palette[c] = "#000000"  # Preto destacado para Outliers
    else:
        palette[c] = colors[color_idx]
        color_idx += 1

sns.scatterplot(
    data=df,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    hue='Cluster',
    style='Status',
    palette=palette,
    s=80,
    alpha=0.85,
    ax=ax
)

ax.set_title("Frequência de Vibração vs Pico de Temperatura", fontsize=14, fontweight='bold')
ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=12)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)

# Ajuste da legenda
plt.legend(title="Agrupamentos", bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

# ---------------------------------------------------------
# 6. Tabela Interativa de Máquinas Anômalas
# ---------------------------------------------------------
st.subheader("🚨 Máquinas Consideradas Anômalas")

df_anomalias = df[df['Cluster'] == -1][['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura']]

if not df_anomalias.empty:
    st.warning(f"Foram encontradas **{len(df_anomalias)}** máquinas operando fora dos padrões normais.")
    
    st.dataframe(
        df_anomalias.style.format({
            'Frequencia_Vibracao': '{:.2f} Hz',
            'Pico_Temperatura': '{:.2f} °C'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("Nenhuma anomalia foi detectada com os parâmetros atuais do DBSCAN.")
