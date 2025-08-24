import streamlit as st
import pandas as pd
import numpy as np
from functions import fetch_prices_yf, to_periodic_returns, fit_capm_all, capm_report, plot_combined_evolution
import plotly.express as px 
import plotly.graph_objects as go

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

st.sidebar.header("⚙️ Configuración del Análisis")

# Elección de Índices
indices_dict = {
    '^GSPC': 'S&P 500 - Estados Unidos',
    '^DJI': 'Dow Jones Industrial Average - Estados Unidos', 
    '^IXIC': 'NASDAQ Composite - Estados Unidos',
    '^URTH': 'MSCI World - Global (Países Desarrollados)',
    '^STOXX50E': 'Euro Stoxx 50 - Zona Euro',
    '^GDAXI': 'DAX - Alemania',
    '^FCHI': 'CAC 40 - Francia',
    '^IBEX': 'IBEX 35 - España',
    '000001.SS': 'Shanghai Composite - China',
    '^MXX': 'S&P/BMV IPC - México',
    '^IPSA': 'IPSA - Chile', 
    '^MERV': 'S&P MERVAL - Argentina',
    '^BVSP': 'IBOVESPA - Brasil',
    '^COLCAP': 'COLCAP - Colombia',
    '^SPBLPGPT': 'S&P/BVL Perú General Index - Perú'
}

nombres_indices = list(indices_dict.values())
tickers_indices = list(indices_dict.keys())

indice_seleccionado_nombre = st.sidebar.selectbox(
    "Selecciona un índice de mercado:",
    options=nombres_indices,
    index=0
)
mkt = tickers_indices[nombres_indices.index(indice_seleccionado_nombre)]

# Inputs de usuario
tickers_input = st.sidebar.text_input('Tickers separados por espacio', 'AAPL MSFT GOOG TSLA')
risk_free_rate = st.sidebar.number_input("Tasa Libre de Riesgo (r)", value=0.02, step=0.01, format="%.3f")

start = st.sidebar.date_input('Fecha de inicio', value=pd.to_datetime('2020-01-01'))
end = st.sidebar.date_input('Fecha de fin', value=pd.to_datetime('today'))

return_method = st.sidebar.radio("Método de cálculo de retornos", ["log"], index=0)

#########################
# Procesamiento Principal
#########################

if st.sidebar.button("🚀 Ejecutar Análisis CAPM", type="primary"):
    
    tickers = tickers_input.split()
    
    if not tickers:
        st.error("Por favor ingresa al menos un ticker")
        st.stop()
    
    with st.spinner("Descargando datos y calculando..."):
        try:
            # Descargar datos de activos
            prices = fetch_prices_yf(tickers, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            returns = to_periodic_returns(prices, method=return_method)
            
            # Descargar datos del mercado
            mkt_prices = fetch_prices_yf([mkt], start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            mkt_returns = to_periodic_returns(mkt_prices, method=return_method).iloc[:, 0]
            
            # Calcular CAPM
            capm_results = fit_capm_all(returns, mkt_returns, risk_free_rate)
            mkt_ret = mkt_returns.mean() * 252
            market_premium = mkt_ret - risk_free_rate

            # Calcular retornos esperados
            expected_returns = risk_free_rate + capm_results['beta'] * [market_premium]
            
            #########################
            # Mostrar Resultados
            #########################
            
            st.success("✅ Análisis completado exitosamente!")
            
            # Gráficos de evolución
            st.header("📈 Evolución de Precios y Rendimientos")
            
            view_option = st.radio(
                "Tipo de visualización:",
                ["Gráfico Combinado"],
                horizontal=True
            )
            
            fig = plot_combined_evolution(prices, returns, "Evolución de Activos")
            st.plotly_chart(fig, use_container_width=True)

            # Resultados CAPM
            st.header("📊 Resultados CAPM")
            
            st.metric("Índice de Mercado", indice_seleccionado_nombre)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_mkt_return = mkt_returns.mean() * 252 if return_method == 'simple' else mkt_returns.mean() * 252
                st.metric("Retorno Mercado", f"{avg_mkt_return:.2%}")
            
            with col2:
                st.metric("Tasa Libre de Riesgo", f"{risk_free_rate:.2%}")
            
            with col3:
                st.metric("Prima de Mercado", f"{market_premium:.2%}")


            # Tabla de resultados detallados
            st.subheader("📋 Parámetros CAPM por Activo")
            
            results_df = capm_results.copy()
            results_df['expected_return'] = expected_returns
            results_df['alpha'] = results_df['alpha'] * 252  # Anualizar alpha
            results_df['expected_return'] = results_df['expected_return']  # Ya está anualizado
            
            # Formatear números
            display_df = results_df[['alpha', 'beta', 'r2', 'n', 'expected_return']].copy()
            display_df['Alpha de Jensen'] = display_df['alpha'].apply(lambda x: f"{x:.4f}")
            display_df['Beta'] = display_df['beta'].apply(lambda x: f"{x:.4f}")
            display_df['R^2'] = display_df['r2'].apply(lambda x: f"{x:.4f}")
            display_df['Retorno Esperado'] = display_df['expected_return'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_df[['Alpha de Jensen', 'Beta', 'R^2', 'n', 'Retorno Esperado']], use_container_width=True)
            
            # Gráfico de Betas
            st.subheader("📊 Betas de los Activos")
            
            fig_beta = px.bar(
                x=results_df.index,
                y=results_df['beta'],
                labels={'x': 'Activo', 'y': 'Beta'},
                title="Betas CAPM por Activo",
                color=results_df['beta'],
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_beta, use_container_width=True)
            
            # Retornos Esperados vs Realizados
            st.subheader("📈 Retornos Esperados vs Realizados")
            
            realized_returns = returns.mean() * 252  # Anualizar
            comparison_df = pd.DataFrame({
                'Realizado': realized_returns,
                'Esperado (CAPM)': expected_returns
            })
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                x=comparison_df.index,
                y=comparison_df['Realizado'],
                name='Retorno Realizado',
                marker_color='lightblue'
            ))
            fig_comparison.add_trace(go.Bar(
                x=comparison_df.index,
                y=comparison_df['Esperado (CAPM)'],
                name='Retorno Esperado CAPM',
                marker_color='lightgreen'
            ))
            
            fig_comparison.update_layout(
                title="Comparación: Retornos Realizados vs Esperados (CAPM)",
                barmode='group',
                xaxis_title="Activo",
                yaxis_title="Retorno Anualizado",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Descargar resultados
            st.download_button(
                label="📥 Descargar Resultados CSV",
                data=results_df.to_csv(),
                file_name="capm_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error en el análisis: {str(e)}")
            st.info("Verifica que los tickers sean válidos y que las fechas tengan datos disponibles")

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
        3. Configura tasa libre de riesgo y prima de mercado
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
            José Cotrina Lejabo
        </a>
    </div>
""", unsafe_allow_html=True)
