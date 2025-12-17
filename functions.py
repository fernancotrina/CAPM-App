import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

def get_stock_prices(tickers, start_date, end_date):
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)["Adj Close"]
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

def returns_minus_rf(returns, rf):
    rf = rf / 252
    returns_reg = returns - rf
    return returns_reg

def capm_regression(returns, market_returns):

    results = {}
    
    for ticker in returns.columns:
        Y = returns[ticker]
        X = sm.add_constant(market_returns)
        model = sm.OLS(Y, X).fit()

        results[ticker] = {
            "Alpha (%)": model.params["const"] * 100,
            "Beta": model.params[market_returns.name],
            "t-Alpha": model.tvalues["const"],
            "t-Beta": model.tvalues[market_returns.name],
            "R2": model.rsquared,
            "N": int(model.nobs)
        }

    return pd.DataFrame(results).T

def finaal_report(returns, market_ticker, rf):
    premium = returns_minus_rf(returns, rf)
    RESULTS = capm_regression(premium.drop(columns=market_ticker), premium[market_ticker])
    RESULTS["Expected Returns (%)"] = (rf + RESULTS["Beta"] * (premium[market_ticker].mean() * 252))*100

    return RESULTS

def plot_stock_prices(prices, title="Precios de las Acciones"):
    fig = go.Figure()

    for stock in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[stock], name=stock, mode='lines'))

    fig.update_layout(title=title, height=400, hovermode='x unified')

    fig.update_xaxes(title_text="Fecha")
    fig.update_yaxes(title_text="Precio ($)")

    return fig

def plot_cumulative_returns(returns, title="Retornos Acumulados (%)"):
    cumulative_returns = (1 + returns).cumprod() * 100

    fig = go.Figure()

    for stock in cumulative_returns.columns:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[stock], name=stock, mode='lines'))

    fig.update_layout(
        title=title,
        height=400,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Fecha")
    fig.update_yaxes(title_text="Retorno Acumulado (%)")

    return fig

def rolling_beta(returns, market_ticker, rf, window):
    # Excess returns
    premium = returns_minus_rf(returns, rf)

    market_excess = premium[market_ticker]
    rolling_betas = pd.DataFrame(index=premium.index[window:])

    for ticker in premium.columns.drop(market_ticker):
        betas = []

        stock_excess = premium[ticker]

        for i in range(window, len(premium)):
            stock_window = stock_excess.iloc[i - window:i]
            market_window = market_excess.iloc[i - window:i]

            if stock_window.isna().any() or market_window.isna().any():
                betas.append(np.nan)
                continue

            X = sm.add_constant(market_window)
            model = sm.OLS(stock_window, X).fit()

            betas.append(model.params[market_excess.name])

        rolling_betas[ticker] = betas

    return rolling_betas.dropna()

def plot_rolling_beta(rolling_betas, title="Evolución de la Beta (Rolling)"):
    fig = go.Figure()

    for stock in rolling_betas.columns:
        fig.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas[stock], mode="lines", name=stock))

    # Línea beta = 1
    fig.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Beta = 1", annotation_position="bottom right")

    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Beta", hovermode="x unified", height=450,
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0.5,
            xanchor="center"
        ),
        title_x=0.5
    )

    return fig
