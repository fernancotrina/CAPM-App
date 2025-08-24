# Import libraries
from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

def fetch_prices_yf(tickers: Iterable[str], start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Descarga precios de yfinance. Devuelve DataFrame (fecha x ticker) de Close ajustado.
    Requiere internet en TU entorno (no en este editor).
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Falta yfinance. Instala con: pip install yfinance") from e

    data = yf.download(list(tickers), start=start, end=end, auto_adjust=auto_adjust, progress=False)
    # Estructura: columnas multiíndice (Adj Close, ...). Tomamos Close/Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        # Prioridad a 'Adj Close' si existe; si no, 'Close'
        if ('Adj Close' in data.columns.get_level_values(0)):
            px = data['Adj Close']
        else:
            px = data['Close']
    else:
        px = data
    return px.dropna(how='all')


def read_prices_csv(path: str, date_col: str = 'Date') -> pd.DataFrame:
    """Lee un CSV con columna de fechas y columnas una por ticker.
    El archivo debe tener formato wide: filas=fecha, columnas=tickers.
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def to_periodic_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Convierte precios a retornos (porcentaje). method: 'simple' o 'log'."""
    prices = prices.sort_index()
    if method == 'simple':
        rets = prices.pct_change()
    elif method == 'log':
        rets = np.log(prices).diff()
    else:
        raise ValueError("method debe ser 'simple' o 'log'")
    return rets.dropna(how='all')


def align_inputs(assets_ret: pd.DataFrame, mkt_ret: pd.Series, rf: Optional[pd.Series | float]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Alinea activos, mercado y rf en el mismo índice temporal.
    Si rf es float, crea una Serie constante (misma frecuencia del mercado)."""
    idx = assets_ret.index
    mkt_ret = pd.Series(mkt_ret, index=mkt_ret.index).dropna()
    common_idx = idx.intersection(mkt_ret.index)
    A = assets_ret.loc[common_idx]
    M = mkt_ret.loc[common_idx]

    if rf is None:
        rf_series = pd.Series(0.0, index=common_idx)
    elif isinstance(rf, (int, float)):
        rf_series = pd.Series(float(rf), index=common_idx)
    else:
        rf_series = pd.Series(rf, index=pd.Series(rf).index).dropna()
        rf_series = rf_series.loc[common_idx]
    return A, M, rf_series

# =============================
# Estimación CAPM
# =============================

@dataclass
class CAPMResult:
    alpha: float
    beta: float
    alpha_t: float
    beta_t: float
    r2: float
    n: int


def fit_capm_ols(asset_excess: pd.Series, mkt_excess: pd.Series) -> CAPMResult:
    """Regresión OLS: Ri - rf = α + β (Rm - rf) + ε"""
    df = pd.concat({'y': asset_excess, 'x': mkt_excess}, axis=1).dropna()
    if df.empty:
        raise ValueError("No hay datos comunes para estimar CAPM")
    X = sm.add_constant(df['x'])
    y = df['y']
    model = sm.OLS(y, X, missing='drop').fit()
    alpha = model.params.get('const', np.nan)
    beta = model.params.get('x', np.nan)
    alpha_t = model.tvalues.get('const', np.nan)
    beta_t = model.tvalues.get('x', np.nan)
    r2 = model.rsquared
    n = int(model.nobs)
    return CAPMResult(alpha, beta, alpha_t, beta_t, r2, n)


def fit_capm_all(assets_ret: pd.DataFrame, mkt_ret: pd.Series, rf: Optional[pd.Series | float] = 0.0) -> pd.DataFrame:
    """Ajusta CAPM para cada activo. Devuelve DataFrame con α, β, t-stats, R², n."""
    A, M, RF = align_inputs(assets_ret, mkt_ret, rf)
    mkt_excess = M - RF
    out = []
    for col in A.columns:
        res = fit_capm_ols(A[col] - RF, mkt_excess)
        out.append({
            'asset': col,
            'alpha': res.alpha,
            'beta': res.beta,
            'alpha_t': res.alpha_t,
            'beta_t': res.beta_t,
            'r2': res.r2,
            'n': res.n,
        })
    return pd.DataFrame(out).set_index('asset')


def rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, rf: Optional[pd.Series | float] = 0.0, window: int = 60) -> pd.Series:
    """β rolling con ventana (p.ej., 60 periodos mensuales ≈ 5 años)."""
    A, M, RF = align_inputs(asset_ret.to_frame('a'), mkt_ret, rf)
    a = (A['a'] - RF)
    m = (M - RF)
    def _beta_win(x):
        y = x.iloc[:, 0]
        x_ = sm.add_constant(x.iloc[:, 1])
        if x_.shape[0] < 3:
            return np.nan
        try:
            return sm.OLS(y, x_, missing='drop').fit().params.get('x', np.nan)
        except Exception:
            return np.nan
    df = pd.concat([a, m], axis=1, keys=['y', 'x']).dropna()
    return df.rolling(window).apply(lambda w: _beta_win(pd.DataFrame(w.reshape(-1,2), columns=['y','x'])), raw=True)

# =============================
# Retornos esperados vía CAPM
# =============================

def capm_expected_returns(betas: pd.Series, market_premium: float, rf: float) -> pd.Series:
    """E[Ri] = rf + βi * (market_premium). market_premium = E[Rm] - rf."""
    return rf + betas * market_premium

# =============================
# Reportes y gráficos
# =============================

def capm_report(assets_ret: pd.DataFrame, mkt_ret: pd.Series, rf: Optional[pd.Series | float] = 0.0,
                market_premium: Optional[float] = None) -> Dict[str, pd.DataFrame | pd.Series]:
    """Genera un pequeño paquete: resultados OLS, betas, expected returns (si se pasa market_premium)."""
    results = fit_capm_all(assets_ret, mkt_ret, rf)
    betas = results['beta']
    out: Dict[str, pd.DataFrame | pd.Series] = {'results': results, 'betas': betas}
    if market_premium is not None and isinstance(rf, (int, float)):
        er = capm_expected_returns(betas, market_premium, float(rf))
        out['expected_returns_capm'] = er
    return out

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# =============================
# Gráficos interactivos con Plotly
# =============================


def plot_combined_evolution(prices_df: pd.DataFrame, returns_df: pd.DataFrame, 
                          main_title: str = "Análisis de Evolución") -> go.Figure:
    """Grafica precios y rendimientos en subplots combinados"""
    # Normalizar precios
    prices = prices_df 
    cumulative_returns = (1 + returns_df).cumprod() - 1
    cumulative_returns *= 100  # Convertir a porcentaje
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Evolución de Precios', 'Retornos Acumulados'),
        vertical_spacing=0.15
    )
    
    # Gráfico de precios (arriba)
    for i, column in enumerate(prices.columns):
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices[column],
                name=column,
                mode='lines',
                showlegend=True,
                hovertemplate='<b>%{x}</b><br>Precio: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Gráfico de rendimientos (abajo)
    for i, column in enumerate(cumulative_returns.columns):
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[column],
                name=column,
                mode='lines',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Retorno: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=dict(text=main_title, x=0.5, xanchor='center', font=dict(size=20)),
        height=700,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Precio de la Acción ($)", row=1, col=1)
    fig.update_yaxes(title_text="Retorno Acumulado (%)", row=2, col=1)
    
    return fig
