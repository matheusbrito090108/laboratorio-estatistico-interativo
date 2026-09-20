import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Sistematização Módulo 3 - ADS", layout="wide")

st.title("Módulo 3: Probabilidade e Simulação")
st.write("Grupo ADS - Demonstração da LGN e do TCL usando Monte Carlo")

# Sidebar
st.sidebar.title("Navegação")
opcao = st.sidebar.radio("Selecione a simulação:", ["Lei dos Grandes Números", "Teorema Central do Limite"])

def render_lgn():
    st.header("Lei dos Grandes Números (LGN)")
    
    col_input, col_graph = st.columns([1, 2])
    
    with col_input:
        p = st.slider("Probabilidade Teórica (p)", 0.1, 0.9, 0.5, 0.05)
        n_trials = st.slider("Total de Lançamentos (N)", 100, 50000, 10000, 500)
    
    with col_graph:
        np.random.seed(42)
        dados = np.random.binomial(1, p, n_trials)
        med_acumulada = np.cumsum(dados) / np.arange(1, n_trials + 1)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(med_acumulada, color='tab:blue', label='Média Amostral')
        ax.axhline(p, color='tab:red', linestyle='--', label=f'Teórico (p={p})')
        ax.set_xlabel("Nº de Ensaios")
        ax.set_ylabel("Frequência Relativa")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    st.write(f"**Resultado final:** Média obtida = {med_acumulada[-1]:.4f} (Esperado = {p})")

def render_tcl():
    st.header("Teorema Central do Limite (TCL)")
    
    col_input, col_graph = st.columns([1, 2])
    
    with col_input:
        dist_tipo = st.selectbox("Distribuição de Origem", ["Uniforme", "Exponencial", "Binomial"])
        tam_amostra = st.slider("Tamanho da Amostra (n)", 2, 100, 30)
        num_simu = st.slider("Número de Amostras (N)", 100, 10000, 2000, 100)
        
    with col_graph:
        np.random.seed(42)
        
        if dist_tipo == "Uniforme":
            amostras = np.random.uniform(0, 1, (num_simu, tam_amostra))
        elif dist_tipo == "Exponencial":
            amostras = np.random.exponential(1, (num_simu, tam_amostra))
        else:
            amostras = np.random.binomial(1, 0.3, (num_simu, tam_amostra))
            
        med_amostras = np.mean(amostras, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(med_amostras, bins=35, density=True, alpha=0.6, color='tab:green', edgecolor='black')
        
        mu, std = norm.fit(med_amostras)
        x = np.linspace(min(med_amostras), max(med_amostras), 100)
        ax.plot(x, norm.pdf(x, mu, std), 'r-', label=f'Ajuste Normal (μ={mu:.2f}, σ={std:.2f})')
        
        ax.set_xlabel("Média Amostral")
        ax.set_ylabel("Densidade")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if opcao == "Lei dos Grandes Números":
    render_lgn()
else:
    render_tcl()
