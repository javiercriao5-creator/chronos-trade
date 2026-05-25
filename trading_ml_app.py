"""
╔══════════════════════════════════════════════════════════════╗
║   SISTEMA DE TRADING CUANTITATIVO CON ML — v1.0              ║
║   Datos reales via yfinance | ML con scikit-learn            ║
║   Basado en principios MiFID y Guía Caja Ingenieros          ║
╚══════════════════════════════════════════════════════════════╝

Para ejecutar:
    pip install -r requirements.txt
    streamlit run trading_ml_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Trading ML — Sistema Cuantitativo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c2333;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #8892a4; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 24px; font-weight: 600; color: #e2e8f0; }
    .metric-pos { color: #48bb78; }
    .metric-neg { color: #fc8181; }
    .signal-buy  { background: #1a3a2a; border: 1px solid #48bb78; border-radius: 8px; padding: 12px; text-align: center; }
    .signal-sell { background: #3a1a1a; border: 1px solid #fc8181; border-radius: 8px; padding: 12px; text-align: center; }
    .signal-hold { background: #2a2a1a; border: 1px solid #ecc94b; border-radius: 8px; padding: 12px; text-align: center; }
    .section-title { font-size: 13px; color: #8892a4; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
    .info-box { background: #1a2332; border-left: 3px solid #4299e1; border-radius: 4px; padding: 12px 16px; font-size: 13px; color: #8892a4; margin: 8px 0; }
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PERFILES INVERSORES (basados en los PDFs)
# ─────────────────────────────────────────────
PERFILES = {
    "Muy conservador": {
        "color": "#48bb78", "rent": "0%", "vol": "0.4%", "var": "-0.3%", "hor": "< 1 año",
        "desc": "Protección del capital. Sin exposición a renta variable.",
        "alloc": {"Liquidez": 60, "Deuda pública < 3a": 40, "Renta fija corp.": 0, "Renta variable": 0, "Alternativos": 0}
    },
    "Conservador": {
        "color": "#4299e1", "rent": "0.4%", "vol": "1.3%", "var": "-0.9%", "hor": "1+ años",
        "desc": "Preservación del capital con rentabilidad moderada.",
        "alloc": {"Liquidez": 20, "Deuda pública < 3a": 38, "Renta fija corp.": 22, "Renta variable": 5, "Alternativos": 15}
    },
    "Moderado": {
        "color": "#ecc94b", "rent": "2.5%", "vol": "7.7%", "var": "-5.2%", "hor": "2+ años",
        "desc": "Rentabilidad a medio plazo. Cartera equilibrada.",
        "alloc": {"Liquidez": 8, "Deuda pública < 3a": 10, "Renta fija corp.": 15, "Renta variable": 52, "Alternativos": 15}
    },
    "Arriesgado": {
        "color": "#ed8936", "rent": "5%", "vol": "14.1%", "var": "-9.5%", "hor": "3+ años",
        "desc": "Retornos significativos. Alta exposición a renta variable.",
        "alloc": {"Liquidez": 3, "Deuda pública < 3a": 0, "Renta fija corp.": 5, "Renta variable": 83, "Alternativos": 9}
    },
    "Muy arriesgado": {
        "color": "#fc8181", "rent": "6.5%", "vol": "19.7%", "var": "-13.2%", "hor": "5+ años",
        "desc": "Máxima rentabilidad a largo plazo. Máxima exposición bursátil.",
        "alloc": {"Liquidez": 1, "Deuda pública < 3a": 0, "Renta fija corp.": 0, "Renta variable": 94, "Alternativos": 5}
    }
}

# Activos por defecto por perfil
ACTIVOS_DEFAULT = {
    "Muy conservador": ["ES0=F", "EWG", "AGG"],
    "Conservador":    ["SPY", "AGG", "GLD"],
    "Moderado":       ["SPY", "QQQ", "GLD", "AAPL"],
    "Arriesgado":     ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"],
    "Muy arriesgado": ["QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN"]
}

# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache 5 minutos
def fetch_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Descarga datos reales de Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        st.warning(f"Error al descargar {symbol}: {e}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos clásicos."""
    df = df.copy()
    # Medias móviles
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_MACD"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal_MACD"]
    # Bollinger Bands
    df["BB_mid"]   = df["Close"].rolling(20).mean()
    std20          = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * std20
    df["BB_lower"] = df["BB_mid"] - 2 * std20
    # Volatilidad realizada (30d)
    df["Ret"]         = df["Close"].pct_change()
    df["Volatility"]  = df["Ret"].rolling(30).std() * np.sqrt(252) * 100
    # Momentum
    df["Momentum_10"] = df["Close"].pct_change(10)
    df["Momentum_20"] = df["Close"].pct_change(20)
    return df

def generate_signal(df: pd.DataFrame) -> dict:
    """
    Genera señal de trading combinando:
    - Cruce de medias (MA20 vs MA50)
    - RSI (sobrecompra/sobreventa)
    - MACD
    - Momentum
    """
    last = df.dropna().iloc[-1]
    score = 0
    reasons = []
    # MA
    if last["MA20"] > last["MA50"]:
        score += 1; reasons.append("MA20 > MA50 (alcista)")
    else:
        score -= 1; reasons.append("MA20 < MA50 (bajista)")
    # RSI
    if last["RSI"] < 35:
        score += 2; reasons.append(f"RSI={last['RSI']:.1f} (sobreventa → oportunidad)")
    elif last["RSI"] > 70:
        score -= 2; reasons.append(f"RSI={last['RSI']:.1f} (sobrecompra → precaución)")
    else:
        reasons.append(f"RSI={last['RSI']:.1f} (neutral)")
    # MACD
    if last["MACD"] > last["Signal_MACD"]:
        score += 1; reasons.append("MACD > Signal (impulso +)")
    else:
        score -= 1; reasons.append("MACD < Signal (impulso -)")
    # Momentum
    if last["Momentum_10"] > 0.02:
        score += 1; reasons.append(f"Momentum 10d={last['Momentum_10']*100:.1f}% (+)")
    elif last["Momentum_10"] < -0.02:
        score -= 1; reasons.append(f"Momentum 10d={last['Momentum_10']*100:.1f}% (-)")
    # Bollinger
    price = last["Close"]
    if price < last["BB_lower"]:
        score += 1; reasons.append("Precio bajo banda inferior (rebote posible)")
    elif price > last["BB_upper"]:
        score -= 1; reasons.append("Precio sobre banda superior (techo posible)")

    if score >= 2:
        signal, color, css = "COMPRAR", "#48bb78", "signal-buy"
    elif score <= -2:
        signal, color, css = "VENDER", "#fc8181", "signal-sell"
    else:
        signal, color, css = "MANTENER", "#ecc94b", "signal-hold"

    return {"signal": signal, "score": score, "color": color, "css": css,
            "reasons": reasons, "last": last}

def ml_prediction(df: pd.DataFrame, days_ahead: int = 15, model_type: str = "Random Forest") -> dict:
    """
    Predicción ML real con features técnicos.
    Modelos: Regresión Lineal o Random Forest.
    """
    df_ml = df.dropna().copy()
    if len(df_ml) < 60:
        return {"error": "Datos insuficientes para ML (mínimo 60 días)"}

    # Features
    features = ["MA20", "MA50", "RSI", "MACD", "Signal_MACD",
                "Volatility", "Momentum_10", "Momentum_20"]
    df_ml["Target"] = df_ml["Close"].shift(-1)
    df_clean = df_ml[features + ["Close", "Target"]].dropna()

    X = df_clean[features].values
    y = df_clean["Target"].values

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()

    model.fit(X_train_s, y_train)
    y_pred_test = model.predict(X_test_s)
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # Predicciones futuras (iterativo)
    future_prices = []
    last_row = df_clean[features].iloc[-1].values.copy()
    last_price = df_clean["Close"].iloc[-1]

    for _ in range(days_ahead):
        x_scaled = scaler.transform(last_row.reshape(1, -1))
        next_price = model.predict(x_scaled)[0]
        future_prices.append(next_price)
        # Actualizar features aproximados (simplificado)
        last_row = last_row.copy()
        last_row[0] = (last_row[0] * 19 + next_price) / 20   # MA20 approx
        last_price = next_price

    # Fechas futuras (días hábiles)
    last_date = df_clean.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days_ahead)

    # Intervalo de confianza (±1 std del error de test)
    err_std = np.std(y_test - y_pred_test)
    ci_upper = [p + 1.5 * err_std for p in future_prices]
    ci_lower = [p - 1.5 * err_std for p in future_prices]

    return {
        "future_dates": future_dates,
        "future_prices": future_prices,
        "ci_upper": ci_upper,
        "ci_lower": ci_lower,
        "mape": mape,
        "model_type": model_type,
        "last_historical_price": df_clean["Close"].iloc[-1],
        "last_historical_date": df_clean.index[-1]
    }

def portfolio_return(symbols: list, weights: list, period: str = "1y") -> dict:
    """Calcula rentabilidad y volatilidad de la cartera."""
    returns = {}
    for s in symbols:
        df = fetch_data(s, period)
        if not df.empty:
            returns[s] = df["Close"].pct_change().dropna()
    if not returns:
        return {}
    ret_df = pd.DataFrame(returns).dropna()
    w = np.array(weights[:len(ret_df.columns)])
    w = w / w.sum()
    port_ret = ret_df.dot(w)
    annual_ret = port_ret.mean() * 252 * 100
    annual_vol = port_ret.std() * np.sqrt(252) * 100
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    cumulative = (1 + port_ret).cumprod()
    return {
        "annual_ret": annual_ret, "annual_vol": annual_vol,
        "sharpe": sharpe, "cumulative": cumulative, "port_ret": port_ret
    }

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    perfil_nombre = st.selectbox(
        "Tu perfil inversor (MiFID)",
        list(PERFILES.keys()),
        index=2,
        help="Basado en el Test de Idoneidad de la Guía Caja Ingenieros"
    )
    perfil = PERFILES[perfil_nombre]

    st.markdown(f"""
    <div style='background:#1c2333; border-left:3px solid {perfil['color']};
         border-radius:4px; padding:10px 14px; margin:8px 0; font-size:13px; color:#8892a4'>
    {perfil['desc']}<br><br>
    📊 Rent. esperada: <strong>{perfil['rent']}</strong><br>
    📉 Volatilidad: <strong>{perfil['vol']}</strong><br>
    ⚠️ VaR 99% mensual: <strong>{perfil['var']}</strong><br>
    🕐 Horizonte mín.: <strong>{perfil['hor']}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Activos a analizar")
    st.caption("Introduce tickers de Yahoo Finance separados por coma")
    default_symbols = ", ".join(ACTIVOS_DEFAULT[perfil_nombre])
    symbols_input = st.text_input("Tickers", value=default_symbols)
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    st.markdown("---")
    periodo = st.select_slider(
        "Período histórico",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        value="1y"
    )
    days_pred = st.slider("Días de predicción", 5, 30, 15)
    model_type = st.radio("Modelo ML", ["Random Forest", "Regresión Lineal"], index=0)

    st.markdown("---")
    st.caption("⚠️ Datos con 15-20 min de retraso. No es asesoramiento financiero.")

# ─────────────────────────────────────────────
# CABECERA PRINCIPAL
# ─────────────────────────────────────────────
col_t, col_p = st.columns([3, 1])
with col_t:
    st.markdown("# 📈 Sistema de Trading Cuantitativo con ML")
    st.caption("Datos reales via Yahoo Finance · ML con scikit-learn · Principios MiFID")
with col_p:
    st.markdown(f"""
    <div style='background:#1c2333; border:1px solid {perfil['color']};
         border-radius:10px; padding:12px; text-align:center; margin-top:10px'>
    <div style='color:#8892a4; font-size:11px'>PERFIL ACTIVO</div>
    <div style='color:{perfil['color']}; font-weight:600; font-size:16px'>{perfil_nombre}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────
if not symbols:
    st.warning("Introduce al menos un ticker en la barra lateral.")
    st.stop()

with st.spinner("Descargando datos de mercado..."):
    dfs = {}
    for s in symbols:
        df = fetch_data(s, periodo)
        if not df.empty:
            dfs[s] = compute_indicators(df)

if not dfs:
    st.error("No se pudieron descargar datos. Verifica los tickers.")
    st.stop()

# ─────────────────────────────────────────────
# SELECTOR DE ACTIVO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("### Análisis por activo")
selected = st.selectbox("Selecciona el activo a analizar en detalle", list(dfs.keys()))
df_sel = dfs[selected]
sig = generate_signal(df_sel)
last = sig["last"]

# ─────────────────────────────────────────────
# MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
price_chg = df_sel["Close"].pct_change().iloc[-1] * 100
price_chg_1m = df_sel["Close"].pct_change(21).iloc[-1] * 100

with c1:
    st.metric("Precio actual", f"{last['Close']:.2f}",
              delta=f"{price_chg:+.2f}% hoy")
with c2:
    st.metric("RSI (14)", f"{last['RSI']:.1f}",
              delta="Sobreventa" if last['RSI'] < 35 else ("Sobrecompra" if last['RSI'] > 70 else "Neutral"))
with c3:
    st.metric("Volatilidad 30d", f"{last['Volatility']:.1f}%")
with c4:
    st.metric("Cambio 1 mes", f"{price_chg_1m:+.2f}%")
with c5:
    st.metric("MACD", f"{last['MACD']:.3f}",
              delta=f"Signal: {last['Signal_MACD']:.3f}")

# ─────────────────────────────────────────────
# SEÑAL DE TRADING
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Señal de trading — ML cuantitativo")

col_sig, col_why = st.columns([1, 2])
with col_sig:
    st.markdown(f"""
    <div class="{sig['css']}" style="padding:20px; text-align:center">
    <div style="font-size:32px; font-weight:700; color:{sig['color']}">{sig['signal']}</div>
    <div style="color:#8892a4; font-size:13px; margin-top:6px">{selected}</div>
    <div style="color:{sig['color']}; font-size:14px; margin-top:4px">Puntuación: {sig['score']:+d}/6</div>
    </div>
    """, unsafe_allow_html=True)

with col_why:
    st.markdown("**¿Por qué esta señal?**")
    for r in sig["reasons"]:
        icon = "✅" if "alcista" in r or "oportunidad" in r or "(+)" in r or "rebote" in r else \
               "❌" if "bajista" in r or "precaución" in r or "(-)" in r or "techo" in r else "⚪"
        st.markdown(f"{icon} {r}")
    st.caption("Combina cruce de medias (MA20/MA50), RSI, MACD, momentum y bandas de Bollinger")

# ─────────────────────────────────────────────
# GRÁFICO PRECIO + INDICADORES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Gráfico de precio e indicadores técnicos")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.04,
    subplot_titles=[f"{selected} — Precio y Medias Móviles", "RSI (14)", "MACD"]
)

# Velas
fig.add_trace(go.Candlestick(
    x=df_sel.index, open=df_sel["Open"], high=df_sel["High"],
    low=df_sel["Low"], close=df_sel["Close"],
    name="Precio", increasing_fillcolor="#48bb78", decreasing_fillcolor="#fc8181",
    increasing_line_color="#48bb78", decreasing_line_color="#fc8181"
), row=1, col=1)

# Medias móviles
for ma, color, label in [("MA20", "#4299e1", "MA 20"),
                          ("MA50", "#ecc94b", "MA 50"),
                          ("MA200", "#a0aec0", "MA 200")]:
    if df_sel[ma].notna().sum() > 5:
        fig.add_trace(go.Scatter(x=df_sel.index, y=df_sel[ma],
                                 line=dict(color=color, width=1.5),
                                 name=label, opacity=0.8), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(
    x=df_sel.index, y=df_sel["BB_upper"], name="BB Upper",
    line=dict(color="#718096", width=1, dash="dot"), opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df_sel.index, y=df_sel["BB_lower"], name="BB Lower",
    line=dict(color="#718096", width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(113,128,150,0.05)", opacity=0.5), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df_sel.index, y=df_sel["RSI"],
                         line=dict(color="#9f7aea", width=2), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="#fc8181", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="#48bb78", row=2, col=1)

# MACD
colors_macd = ["#48bb78" if v >= 0 else "#fc8181" for v in df_sel["MACD_hist"].fillna(0)]
fig.add_trace(go.Bar(x=df_sel.index, y=df_sel["MACD_hist"],
                     marker_color=colors_macd, name="Histograma MACD", opacity=0.7), row=3, col=1)
fig.add_trace(go.Scatter(x=df_sel.index, y=df_sel["MACD"],
                         line=dict(color="#4299e1", width=1.5), name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df_sel.index, y=df_sel["Signal_MACD"],
                         line=dict(color="#fc8181", width=1.5), name="Signal"), row=3, col=1)

fig.update_layout(
    height=700, template="plotly_dark", showlegend=True,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", y=1.02, x=0)
)
fig.update_xaxes(showgrid=True, gridcolor="#2d3748")
fig.update_yaxes(showgrid=True, gridcolor="#2d3748")

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PREDICCIÓN ML
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Predicción ML — {model_type} ({days_pred} días)")

with st.spinner(f"Entrenando modelo {model_type}..."):
    pred_result = ml_prediction(df_sel, days_pred, model_type)

if "error" in pred_result:
    st.warning(pred_result["error"])
else:
    mape = pred_result["mape"]
    last_hist_price = pred_result["last_historical_price"]
    predicted_end = pred_result["future_prices"][-1]
    change_pct = (predicted_end - last_hist_price) / last_hist_price * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Error del modelo (MAPE)", f"{mape:.2f}%",
                  help="Mean Absolute Percentage Error en datos de test (20% final)")
    with c2:
        st.metric(f"Precio predicho en {days_pred}d", f"{predicted_end:.2f}")
    with c3:
        st.metric("Variación esperada", f"{change_pct:+.2f}%",
                  delta=f"Desde {last_hist_price:.2f}")

    # Gráfico de predicción
    hist_recent = df_sel["Close"].iloc[-60:]
    fig_pred = go.Figure()

    # Histórico reciente
    fig_pred.add_trace(go.Scatter(
        x=hist_recent.index, y=hist_recent.values,
        line=dict(color="#4299e1", width=2.5), name="Precio histórico"
    ))

    # Conexión
    fig_pred.add_trace(go.Scatter(
        x=[pred_result["last_historical_date"], pred_result["future_dates"][0]],
        y=[pred_result["last_historical_price"], pred_result["future_prices"][0]],
        line=dict(color="#48bb78", width=2, dash="dot"), showlegend=False
    ))

    # Intervalo de confianza
    fig_pred.add_trace(go.Scatter(
        x=list(pred_result["future_dates"]) + list(pred_result["future_dates"])[::-1],
        y=pred_result["ci_upper"] + pred_result["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(72,187,120,0.1)",
        line=dict(color="rgba(72,187,120,0)"), name="Intervalo de confianza"
    ))

    # Predicción
    fig_pred.add_trace(go.Scatter(
        x=pred_result["future_dates"], y=pred_result["future_prices"],
        line=dict(color="#48bb78", width=2.5, dash="dash"), name=f"Predicción {model_type}",
        mode="lines+markers", marker=dict(size=5)
    ))

    fig_pred.update_layout(
        height=380, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=True, gridcolor="#2d3748"),
        yaxis=dict(showgrid=True, gridcolor="#2d3748"),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.info(f"**¿Cómo funciona?** El modelo {model_type} aprende de 8 indicadores técnicos "
            f"(MA20, MA50, RSI, MACD, Volatilidad, Momentum 10d y 20d) para predecir el precio del día siguiente. "
            f"Error promedio en datos nuevos: {mape:.1f}%. A menor MAPE, más fiable la predicción.")

# ─────────────────────────────────────────────
# RESUMEN DE TODOS LOS ACTIVOS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Resumen de señales — todos los activos")

cols = st.columns(len(dfs))
for i, (sym, df_a) in enumerate(dfs.items()):
    sig_a = generate_signal(df_a)
    last_a = sig_a["last"]
    chg_a = df_a["Close"].pct_change().iloc[-1] * 100
    chg_1m_a = df_a["Close"].pct_change(21).iloc[-1] * 100
    with cols[i]:
        st.markdown(f"""
        <div class="{sig_a['css']}" style="margin-bottom:8px">
        <div style="font-weight:600; font-size:14px; color:#e2e8f0">{sym}</div>
        <div style="font-size:18px; font-weight:600; color:{sig_a['color']}">{sig_a['signal']}</div>
        <div style="font-size:13px; color:#8892a4">{last_a['Close']:.2f}</div>
        <div style="font-size:12px; color:{'#48bb78' if chg_a >= 0 else '#fc8181'}">
            {'▲' if chg_a >= 0 else '▼'} {abs(chg_a):.2f}% hoy
        </div>
        <div style="font-size:11px; color:#718096; margin-top:4px">
            1m: {chg_1m_a:+.2f}% | RSI: {last_a['RSI']:.0f}
        </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ASIGNACIÓN DE CARTERA
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Asignación de cartera — Perfil {perfil_nombre}")
st.caption("Basado en la Strategic Asset Allocation (SAA) de la Guía Caja Ingenieros y normativa MiFID")

col_pie, col_bar = st.columns(2)

alloc = {k: v for k, v in perfil["alloc"].items() if v > 0}

with col_pie:
    colors_alloc = ["#1D9E75", "#5DCAA5", "#9FE1CB", "#378ADD", "#85B7EB",
                    "#EF9F27", "#D4537E", "#E24B4A", "#639922"]
    fig_pie = go.Figure(go.Pie(
        labels=list(alloc.keys()), values=list(alloc.values()),
        hole=0.55, marker_colors=colors_alloc[:len(alloc)],
        textinfo="label+percent", textfont_size=12,
        hovertemplate="%{label}: %{value}%<extra></extra>"
    ))
    fig_pie.update_layout(
        height=300, template="plotly_dark",
        paper_bgcolor="#0e1117", showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        annotations=[dict(text=perfil_nombre[:10], x=0.5, y=0.5,
                          font_size=13, showarrow=False, font_color="#8892a4")]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    fig_bar = go.Figure(go.Bar(
        x=list(alloc.values()), y=list(alloc.keys()),
        orientation="h",
        marker_color=colors_alloc[:len(alloc)],
        text=[f"{v}%" for v in alloc.values()],
        textposition="auto"
    ))
    fig_bar.update_layout(
        height=300, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=True, gridcolor="#2d3748", title="Peso (%)"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# RENDIMIENTO DE CARTERA (si hay 2+ activos)
# ─────────────────────────────────────────────
if len(dfs) >= 2:
    st.markdown("---")
    st.markdown("### Rendimiento de la cartera (igual ponderación)")

    with st.spinner("Calculando rendimiento de la cartera..."):
        n = len(dfs)
        weights = [1/n] * n
        port = portfolio_return(list(dfs.keys()), weights, periodo)

    if port:
        c1, c2, c3 = st.columns(3)
        with c1:
            color = "normal" if port["annual_ret"] >= 0 else "inverse"
            st.metric("Rentabilidad anual (cartera)", f"{port['annual_ret']:.2f}%")
        with c2:
            st.metric("Volatilidad anual (cartera)", f"{port['annual_vol']:.2f}%")
        with c3:
            st.metric("Ratio Sharpe", f"{port['sharpe']:.2f}",
                      help="Rentabilidad ajustada por riesgo. >1 = bueno, >2 = excelente")

        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=port["cumulative"].index, y=port["cumulative"].values,
            fill="tozeroy", line=dict(color="#4299e1", width=2),
            fillcolor="rgba(66,153,225,0.1)", name="Cartera (igual ponderación)"
        ))
        fig_port.add_hline(y=1, line_dash="dot", line_color="#718096")
        fig_port.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(showgrid=True, gridcolor="#2d3748"),
            yaxis=dict(showgrid=True, gridcolor="#2d3748",
                       title="Crecimiento €1 invertido"),
            showlegend=False
        )
        st.plotly_chart(fig_port, use_container_width=True)

# ─────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:12px; padding:10px 0'>
⚠️ Este sistema es educativo y de investigación. No constituye asesoramiento financiero.<br>
Basado en principios de la Guía de Inversiones de Caja Ingenieros y la normativa MiFID.<br>
Datos con retraso de 15-20 min via Yahoo Finance.
</div>
""", unsafe_allow_html=True)
