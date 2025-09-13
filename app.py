import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Carregar o modelo ---
@st.cache_resource
def load_model():
    """
    Carrega o modelo de regressão logística do arquivo .pkl.
    """
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Erro: O arquivo 'logistic_regression_model.pkl' não foi encontrado. Por favor, certifique-se de que ele está na mesma pasta do seu aplicativo Streamlit.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

# Carregar o modelo na inicialização do aplicativo
model = load_model()

# --- Configuração da página ---
st.set_page_config(
    page_title="📊 Demonstração do Modelo de Machine Learning",
    page_icon="🧠",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1a73e8;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 20px;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f1f3f4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        width: 45%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 1.1rem;
        color: #5f6368;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Título e introdução ---
st.markdown('<h1 class="main-header">Modelo 1 - Classificação - Preços fora do Padrão</h1>', unsafe_allow_html=True)
st.markdown('**Preço (price):** Insira o valor do produto que você quer verificar.')
st.markdown('**Relação de Preço (price_ratio_cat):** Este campo é crucial para o modelo. Ele mostra a relação entre o preço do produto e o preço médio de mercado. Por exemplo:')
st.markdown('Um valor como 1.5 significa que o preço atual é 50% maior que o preço médio.')
st.markdown('Um valor como 0.8 significa que o preço atual é 20% menor que o preço médio.')
st.markdown('O modelo usa essa informação para entender se o preço está fora do padrão, independentemente do seu valor absoluto.')

# --- Formulário de entrada de dados com valores pré-preenchidos ---
st.subheader("Simulador de Previsão")
st.markdown("`X.shape` indica que seu modelo trabalha com 2 features. Você pode ajustar os valores abaixo:")

# Criar colunas para os inputs
col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input(
        "Preço (Ex: 735 a 10000)",
        min_value=0.0,
        max_value=10000.0,
        value=735.0,
        step=0.1
    )

with col2:
    feature_2 = st.number_input(
        "Relação de Preço (price_ratio_cat) (Ex: 1,48 a 100)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

# --- Botão de previsão ---
if st.button("🚀 Gerar Previsão", type="primary", use_container_width=True):
    if model:
        # Preparar os dados de entrada
        input_data = np.array([[feature_1, feature_2]])
        
        # Realizar a previsão
        with st.spinner("Analisando os dados..."):
            try:
                # Previsão da classe (0 ou 1)
                prediction = model.predict(input_data)[0]
                
                # Previsão da probabilidade (confiança)
                probability = model.predict_proba(input_data)[0]
                confidence = probability[prediction] * 100
                
                # Exibir os resultados
                st.success("✅ Previsão Gerada!")
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                
                # Classe Prevista
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">Classe Prevista</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format("0" if prediction == 0 else "1"), unsafe_allow_html=True)

                # Confiança da Previsão
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-title">Confiança da Previsão</div>
                    <div class="metric-value">{:.1f}%</div>
                </div>
                """.format(confidence), unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detalhes da previsão
                st.subheader("Detalhes da Previsão")
                st.info(f"O modelo previu a **classe {prediction}** com **{confidence:.1f}% de confiança.**")
                
                # Explicação para o usuário
                if prediction == 1:
                    st.markdown("Isso significa que, com base nos valores fornecidos, o modelo **classificou os dados na classe 1**.")
                else:
                    st.markdown("Isso significa que, com base nos valores fornecidos, o modelo **classificou os dados na classe 0**.")
            
            except Exception as e:
                st.error(f"Ocorreu um erro durante a previsão: {e}")
    else:
        st.warning("⚠️ O modelo não pôde ser carregado. Verifique o console para mais detalhes.")
