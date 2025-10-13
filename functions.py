import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except:
    pass

def get_stock_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    else:
        prices = data
    return prices.dropna()

def calculate_returns(prices, return_type='simple'):
    prices = prices.sort_index()
    if return_type == 'Simple':
        returns = prices.pct_change()
    elif return_type == 'Log':
        returns = np.log(prices).diff()
    else:
        raise ValueError("Usa 'Simpe' o 'Log' para el Tipo de Cálculo de Retornos")
    return returns.dropna()

def prepare_data(stock_returns, market_returns, risk_free_rate=0.0):
    # Retonro de mercado
    market_returns = pd.Series(market_returns).dropna()

    # Retorno de acciones
    common_dates = stock_returns.index.intersection(market_returns.index)
    stocks_aligned = stock_returns.loc[common_dates]
    market_aligned = market_returns.loc[common_dates]

    # Tasa libre de riesgo — convertir anual a diaria
    rf_daily = float(risk_free_rate) / (252)
    rf_series = pd.Series(rf_daily, index=common_dates)

    return stocks_aligned, market_aligned, rf_series

class CAPMResults:
    def __init__(self, alpha, beta, alpha_t, beta_t, r_squared, num_obs):
        self.alpha = alpha
        self.beta = beta
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.r2 = r_squared
        self.n = num_obs

def run_capm_regression(stock_excess_returns, market_excess_returns):
    data = pd.DataFrame({'stock': stock_excess_returns, 'market': market_excess_returns}).dropna()
    if data.empty:
        raise ValueError("No hay datos suficientes para la regresión")
    X = sm.add_constant(data['market'])
    y = data['stock']
    model = sm.OLS(y, X).fit()
    return CAPMResults(
        model.params['const'],
        model.params['market'],
        model.tvalues['const'],
        model.tvalues['market'],
        model.rsquared,
        len(data)
    )

def calculate_all_capm(stock_returns, market_returns, risk_free=0.0):
    """
    Calcula CAPM para múltiples acciones y devuelve resultados anualizados
    """
    stocks, market, rf = prepare_data(stock_returns, market_returns, risk_free)
    market_excess = market - rf
    
    results = []
    
    for stock_name in stocks.columns:
        stock_excess = stocks[stock_name] - rf
        capm_result = run_capm_regression(stock_excess, market_excess)
        
        # Calcular retorno esperado usando CAPM (CORREGIDO)
        market_premium_anual = market_excess.mean() * 252
        expected_return = (rf.mean() * 252) + (capm_result.beta * market_premium_anual)
        
        # Alpha anualizado CORREGIDO - ya está en términos diarios, multiplicar por 252
        alpha_anual = capm_result.alpha * 252
        
        results.append({
            'Stock': stock_name,
            'alpha': alpha_anual * 100,  # Convertir a porcentaje
            'beta': capm_result.beta,
            'R_squared': capm_result.r2 * 100,  # R² en porcentaje
            'Observaciones': capm_result.n,
            'Retorno_Esperado': expected_return * 100,  # Convertir a porcentaje
            'T-Alpha': capm_result.alpha_t,
            'T-Beta': capm_result.beta_t
        })
    
    return pd.DataFrame(results).set_index('Stock')

def rolling_beta_analysis(stock_returns, market_returns, window=60, risk_free=0.0):
    stocks, market, rf = prepare_data(stock_returns, market_returns, risk_free)
    market_excess = market - rf
    rolling_betas = {}
    for stock in stocks.columns:
        stock_excess = stocks[stock] - rf
        betas = []
        for i in range(window, len(stock_excess)):
            stock_window = stock_excess.iloc[i - window:i]
            market_window = market_excess.iloc[i - window:i]
            if stock_window.isna().any() or market_window.isna().any():
                betas.append(np.nan)
                continue
            X = sm.add_constant(market_window)
            model = sm.OLS(stock_window, X).fit()
            betas.append(model.params.iloc[1])  # En lugar de model.params[1]
        dates = stock_excess.index[window:]
        rolling_betas[stock] = pd.Series(betas, index=dates)
    return rolling_betas

def capm_expected_return(beta, market_return, risk_free_rate):
    return risk_free_rate + beta * (market_return - risk_free_rate)

def create_capm_report(stock_returns, market_returns, risk_free=0.0, expected_market_return=None):
    capm_results = calculate_all_capm(stock_returns, market_returns, risk_free)
    betas = capm_results['beta']
    report = {'capm_results': capm_results, 'betas': betas}
    if expected_market_return is not None and isinstance(risk_free, (int, float)):
        expected_returns = betas.apply(lambda beta: capm_expected_return(beta, expected_market_return, risk_free))
        report['expected_returns'] = expected_returns
    return report

def plot_stock_analysis(prices, returns, title="Análisis de Acciones"):
    cumulative_returns = (1 + returns).cumprod() - 1
    cumulative_returns *= 100
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Precios de las Acciones', 'Retornos Acumulados (%)'),
                        vertical_spacing=0.2)
    for stock in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[stock], name=stock, mode='lines'), row=1, col=1)
    for stock in cumulative_returns.columns:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[stock], name=stock, mode='lines'),
                      row=2, col=1)
    fig.update_layout(title=title, height=600, hovermode='x unified')
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Retorno Acumulado (%)", row=2, col=1)
    return fig

def plot_rolling_beta(rolling_betas_dict):
    fig = go.Figure()
    for stock, beta_series in rolling_betas_dict.items():
        fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series.values, mode='lines', name=stock))
    fig.add_hline(y=1, line_dash="dot", line_color="gray",
                  annotation_text="Beta = 1", annotation_position="bottom right")
    fig.update_layout(title="Evolución de la Beta (Rolling)",
                      xaxis_title="Fecha", yaxis_title="Beta",
                      template="plotly_white", hovermode="x unified",
                      height=450, legend=dict(orientation="h", y=-0.2, x=0.3), title_x=0.5)
    return fig
