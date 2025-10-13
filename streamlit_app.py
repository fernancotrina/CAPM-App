import streamlit as st
import pandas as pd
from functions import (
    get_stock_prices,
    calculate_returns,
    create_capm_report,
    rolling_beta_analysis,
    plot_stock_analysis,
    plot_rolling_beta
)


# Page configuration
st.set_page_config(
    page_title="Capital Asset Pricing Model",
    page_icon="📊",
    layout="wide",
)

#########################
# Encabezado
#########################

st.markdown("""
    <style>
    .main-title {
        font-family: 'Arial', sans-serif;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin: 2rem 0;
        padding: 15px;
    }
    
    .author-simple {
        text-align: center;
        margin: 1.5rem 0;
        font-family: 'Arial', sans-serif;
    }
    
    .author-link-simple {
        text-decoration: none;
        color: #333 !important;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .linkedin-icon-simple {
        width: 22px;
        height: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
    <div class="main-title">
        Capital Asset Pricing Model (CAPM)
    </div>
""", unsafe_allow_html=True)

#########################
# Inputs
#########################

st.title("📈 Análisis CAPM y Beta Rolling")
st.write("Analiza acciones y sus betas frente al mercado usando datos de Yahoo Finance.")

# Sidebar
st.sidebar.header("⚙️ Configuración del Análisis")

    # Elección de Índices
indices_dict = {
    '^GSPC': 'S&P 500 - Estados Unidos',
    '^DJI': 'Dow Jones Industrial Average - Estados Unidos', 
    '^IXIC': 'NASDAQ Composite - Estados Unidos',
    'URTH': 'MSCI World - Global (Países Desarrollados)',
    '^STOXX50E': 'Euro Stoxx 50 - Zona Euro',
    '^GDAXI': 'DAX - Alemania',
    '^FCHI': 'CAC 40 - Francia',
    '^IBEX': 'IBEX 35 - España',
    '000001.SS': 'Shanghai Composite - China',
    '^MXX': 'S&P/BMV IPC - México',
    '^MERV': 'S&P MERVAL - Argentina',
    '^SPBLPGPT': 'S&P/BVL Perú General Index - Perú'
}

nombres_indices = list(indices_dict.values())
tickers_indices = list(indices_dict.keys())

indice_seleccionado_nombre = st.sidebar.selectbox(
    "Selecciona un índice de mercado:",
    options=nombres_indices,
    index=0
)
market_ticker = tickers_indices[nombres_indices.index(indice_seleccionado_nombre)]

    # Elección de Tickers
tickers_input = st.sidebar.text_input("Tickers separados por espacio", "AAPL MSFT GOOG")

    # Elección de Fechas
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", value=pd.to_datetime("today"))

    # Elección de Tasa Libre de Riesgo
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo (%)", value=3.00) / 100

    # Elección de Método de Retornos y Ventana Rolling
return_type = st.sidebar.selectbox("Método de cálculo de retornos", ["Simple", "Log"])
window = st.sidebar.number_input("Ventana Rolling (días)", 30, 250, 60)

#########################
# Procesamiento Principal
#########################

if st.sidebar.button("🚀 Ejecutar Análisis CAPM", type="primary"):
    tickers = [t.strip().upper() for t in tickers_input.split()]
    all_tickers = tickers + [market_ticker]

    prices = get_stock_prices(all_tickers, start_date, end_date)

    stock_prices = prices[tickers]
    market_prices = prices[market_ticker]

    stock_returns = calculate_returns(stock_prices, return_type)
    market_returns = calculate_returns(market_prices, return_type)

    avg_mkt_ret = market_returns.mean() * 252
    market_premium = avg_mkt_ret - risk_free_rate

    st.success("✅ Análisis completado exitosamente!")

    #########################
    # Mostrar Resultados
    #########################

    st.subheader("📉 Gráfico de Precios y Retornos")
    st.plotly_chart(plot_stock_analysis(stock_prices, stock_returns), use_container_width=True)

    st.subheader("📉 Resultados del Modelo CAPM")

    st.metric("Índice de Mercado", indice_seleccionado_nombre)
            
    col1, col2, col3 = st.columns(3)
            
    with col1:
        st.metric("Retorno Mercado", f"{avg_mkt_ret:.2%}")
            
    with col2:
        st.metric("Tasa Libre de Riesgo", f"{risk_free_rate:.2%}")
            
    with col3:
        st.metric("Prima de Mercado", f"{market_premium:.2%}")

    st.subheader("📋 Parámetros CAPM por Activo")
    
    report = create_capm_report(stock_returns, market_returns, risk_free_rate)
    st.dataframe(report["capm_results"])

    st.subheader("📉 Betas Rolling")
    rolling_betas = rolling_beta_analysis(stock_returns, market_returns, window=window, risk_free=risk_free_rate)
    st.plotly_chart(plot_rolling_beta(rolling_betas), use_container_width=True)

#########################
# Pantalla Principal
#########################


else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Ejecutar Análisis CAPM' para comenzar")
    
    # Información de ejemplo
    st.markdown("---")
    st.subheader("ℹ️ Instrucciones de Uso")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📋 Pasos:**
        1. Ingresa los tickers separados por espacio
        2. Selecciona el índice de mercado de referencia
        3. Configura tasa libre de riesgo
        4. Selecciona el rango de fechas
        5. Haz clic en 'Ejecutar Análisis CAPM'
        """)
    
    with col2:
        st.markdown("""
        **🎯 Ejemplos de Tickers:**
        - **Acciones US:** AAPL, MSFT, GOOG, TSLA, AMZN
        - **ETFs:** SPY, QQQ, VTI, VOO
        - **Mercados:** ^GSPC (S&P500), ^IXIC (NASDAQ)
        """)
    
    st.markdown("---")


# Información del autor
st.markdown("""
    <div class="author-simple">
        <span style="color: #666; font-size: 1rem;">
            Created by:
        </span>
        <br>
        <a href="https://www.linkedin.com/in/fcotrina/" target="_blank" class="author-link-simple">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" class="linkedin-icon-simple">
            Fernando Cotrina
        </a>
    </div>
""", unsafe_allow_html=True)
