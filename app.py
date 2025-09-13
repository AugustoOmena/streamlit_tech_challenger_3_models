import streamlit as st
import pickle
import numpy as np

# --- Carregar o modelo ---
@st.cache_resource
def load_model():
    """Carrega o modelo de regressão logística a partir de um arquivo pickle."""
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Erro: O arquivo 'logistic_regression_model.pkl' não foi encontrado. "
                 "Certifique-se de que ele está na mesma pasta do aplicativo Streamlit.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

# --- Função Auxiliar para Exibição (COM A CORREÇÃO) ---
def exibir_painel_de_resultado(class_text, confidence=None, color=None):
    """
    Renderiza o painel de resultados na interface.
    O painel exibe a classe prevista e, opcionalmente, a confiança da previsão.
    
    Args:
        class_text (str): O texto da classe a ser exibido.
        confidence (float, optional): O valor da confiança da previsão. Defaults to None.
        color (str, optional): A cor do texto da confiança. Defaults to None.
    """
    confidence_html = ""
    if confidence is not None and color is not None:
        confidence_html = f'<h3 style="margin-top:20px; color:white;">Confiança da Previsão</h3><p style="font-size:40px; font-weight:bold; color: {color}; margin:0;">{confidence:.1f}%</p>'

    st.markdown(f"""
    <div style="
        border-radius: 15px;
        padding: 20px;
        background-color: rgb(28, 31, 37);
        margin-bottom: 20px;
    ">
        <h2 style="margin-bottom: 15px; color:white;">Classe Prevista</h2>
        <p style="font-size:25px; font-weight:bold; margin:0; color:white;">{class_text}</p>
        {confidence_html}
    </div>
    """, unsafe_allow_html=True)


# --- Início da Aplicação ---

model = load_model()

# --- Configuração da página ---
st.set_page_config(
    page_title="Tech Challenger 3",
    page_icon="📊",
    layout="wide"
)

# --- Layout principal ---
left_col, spacer, right_col = st.columns([1.2, 0.2, 1])

with left_col:
    st.markdown('<h1 style="color:#1a73e8;">Modelo 1 - Classificação - Preços fora do Padrão</h1>', unsafe_allow_html=True)

    st.markdown("""
    **Preço (price):** Insira o valor do produto que você quer verificar.  
    **Relação de Preço (price_ratio_cat):** Este campo é crucial para o modelo.  

    - Um valor como `1.5` significa que o preço atual é **50% maior** que o preço médio.  
    - Um valor como `0.8` significa que o preço atual é **20% menor** que o preço médio.  

    O modelo usa essa informação para entender se o preço está fora do padrão,
    independentemente do valor absoluto.
    """)

    c1, c2 = st.columns(2)
    with c1:
        feature_1 = st.number_input(
            "Preço (Ex: 735 a 10000)",
            min_value=0.0,
            max_value=10000.0,
            value=735.0,
            step=0.1
        )
    with c2:
        feature_2 = st.number_input(
            "Relação de Preço (price_ratio_cat) (Ex: 1,48 a 100)",
            min_value=0.0,
            max_value=100.0,
            value=1.48,
            step=0.1
        )

    gerar = st.button("🚀 Classificar", type="primary", use_container_width=True)

with right_col:
    st.markdown('<div style="margin-top:50px;"></div>', unsafe_allow_html=True)

    # --- Lógica de Classificação e Exibição ---
    
    # Valores padrão para exibição inicial
    class_text_result = "Ainda não classificado"
    confidence_result = None
    color_result = None

    # Se o botão for pressionado, executa a predição e atualiza os valores de exibição
    if gerar:
        if model:
            input_data = np.array([[feature_1, feature_2]])
            with st.spinner("Analisando os dados..."):
                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    confidence_result = probability[prediction] * 100

                    if prediction == 1:
                        class_text_result = "🔴 Preço fora do padrão"
                        color_result = "#e53935"
                    else:
                        class_text_result = "✅ Preço normal"
                        color_result = "#43a047"

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a previsão: {e}")
                    class_text_result = "Erro na classificação"

        else:
            st.warning("⚠️ O modelo não pôde ser carregado. Verifique o console para mais detalhes.")


    exibir_painel_de_resultado(class_text_result, confidence_result, color_result)