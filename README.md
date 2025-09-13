# Tech Challenger 3

Um aplicativo web simples construído com Streamlit para classificar se o preço de um produto está dentro ou fora do padrão de mercado, utilizando um modelo de machine learning pré-treinado.

<img width="2541" height="1427" alt="image" src="https://github.com/user-attachments/assets/be45b73c-2d1d-4c9d-8b6e-c77a6d180b4b" />


## 🎯 Sobre o Projeto

Este projeto fornece uma interface amigável para interagir com um modelo de regressão logística. O objetivo é permitir que um usuário insira duas informações-chave de um produto — seu preço e sua relação com o preço médio da categoria — e receba uma classificação instantânea sobre a normalidade desse valor.

## ✨ Funcionalidades

-   **Interface Intuitiva:** Campos de entrada claros para o usuário.
-   **Classificação em Tempo Real:** Pressione o botão "Classificar" para obter um resultado instantâneo.
-   **Exibição de Confiança:** Além da classe ("Preço Normal" ou "Fora do Padrão"), o app exibe a confiança percentual do modelo na previsão.

## 🛠️ Tecnologias Utilizadas

-   **Python**
-   **Streamlit** - Para a criação da interface web.
-   **Scikit-learn** - Para carregar e utilizar o modelo de machine learning.
-   **Numpy** - Para a manipulação de dados de entrada.

## 🚀 Como Executar

Siga os passos abaixo para rodar o projeto localmente.

### Pré-requisitos

-   Python 3.8 ou superior
-   `pip` (gerenciador de pacotes do Python)
-   O arquivo do modelo `logistic_regression_model.pkl` deve estar na mesma pasta que o script principal.

### Instalação e Execução

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-seu-repositorio>
    cd <nome-do-seu-repositorio>
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo:
    ```txt
    streamlit
    numpy
    scikit-learn
    ```
    E então instale-as:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Rode o aplicativo Streamlit:**
    (Supondo que seu arquivo Python se chame `app.py`)
    ```bash
    streamlit run app.py
    ```

Pronto! O aplicativo abrirá automaticamente no seu navegador.
