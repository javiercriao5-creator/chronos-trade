# 📈 Sistema de Trading Cuantitativo con ML

> Aplicación interactiva de análisis bursátil y predicción con Machine Learning, construida con Python y Streamlit. Datos reales via Yahoo Finance. Principios de inversión basados en la normativa MiFID y la Guía de Inversiones de Caja Ingenieros.

---

## 🧠 ¿Qué hace esta app?

Este sistema combina **análisis técnico cuantitativo** con **modelos de Machine Learning** para ayudar a cualquier persona — sin conocimientos previos de trading — a entender el estado de un activo financiero y tomar decisiones informadas.

### Flujo principal

```
Datos reales (yfinance)
        ↓
Indicadores técnicos (RSI, MACD, MA, Bollinger, Momentum)
        ↓
Modelo ML (Random Forest / Regresión Lineal)
        ↓
Señal de trading (COMPRAR / MANTENER / VENDER)
+ Predicción de precio (N días) con intervalo de confianza
+ Asignación de cartera según perfil MiFID
```

---

## 🚀 Instalación y uso

### Requisitos

- Python 3.9 o superior
- pip

### Pasos

```bash
# 1. Clona el repositorio
git clone https://github.com/TU_USUARIO/trading-ml-sistema.git
cd trading-ml-sistema

# 2. (Opcional) Crea un entorno virtual
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Arranca la app
streamlit run trading_ml_app.py
```

La app se abrirá automáticamente en `http://localhost:8501`.

---

## ✨ Funcionalidades

### 📊 Perfil inversor (normativa MiFID)
Elige tu perfil entre los 5 definidos por la Guía de Inversiones de Caja Ingenieros:

| Perfil | Rent. esperada | Volatilidad | VaR 99% mensual | Horizonte |
|---|---|---|---|---|
| Muy conservador | 0% | 0.4% | -0.3% | < 1 año |
| Conservador | 0.4% | 1.3% | -0.9% | 1+ años |
| Moderado | 2.5% | 7.7% | -5.2% | 2+ años |
| Arriesgado | 5.0% | 14.1% | -9.5% | 3+ años |
| Muy arriesgado | 6.5% | 19.7% | -13.2% | 5+ años |

### 📡 Datos de mercado reales
- Descarga automática via **Yahoo Finance** (`yfinance`)
- Compatible con cualquier ticker: acciones, ETFs, índices, criptomonedas, divisas
- Ejemplos: `AAPL`, `NVDA`, `SAN.MC`, `^IBEX`, `BTC-USD`, `GLD`, `SPY`
- Períodos configurables: 3 meses → 5 años
- Cache de 5 minutos para no saturar la API

### 📐 Indicadores técnicos calculados
| Indicador | Descripción |
|---|---|
| MA20 / MA50 / MA200 | Medias móviles simples |
| RSI (14) | Índice de fuerza relativa |
| MACD | Convergencia/divergencia de medias exponenciales |
| Bandas de Bollinger | Volatilidad y canales de precio |
| Volatilidad realizada (30d) | Desviación estándar anualizada |
| Momentum (10d / 20d) | Variación relativa de precio |

### 🤖 Modelos de Machine Learning
- **Random Forest Regressor** — ensemble de 100 árboles, robusto a ruido de mercado
- **Regresión Lineal** — baseline interpretable
- Entrenamiento automático sobre el 80% de los datos históricos
- Evaluación con **MAPE** sobre el 20% de test
- Predicción iterativa de N días con **intervalo de confianza (±1.5σ)**
- Features: 8 indicadores técnicos estandarizados con `StandardScaler`

### 🔔 Señales de trading cuantitativas
El sistema puntúa cada activo de -6 a +6 combinando 5 criterios:

```
✅ MA20 > MA50          → +1  (tendencia alcista)
✅ RSI < 35             → +2  (sobreventa: oportunidad)
✅ MACD > Signal        → +1  (impulso positivo)
✅ Momentum 10d > 2%    → +1  (precio subiendo)
✅ Precio < BB inferior → +1  (posible rebote)
```

- Puntuación ≥ +2 → **COMPRAR**
- Puntuación ≤ -2 → **VENDER**
- Resto          → **MANTENER**

### 💼 Análisis de cartera
- Rendimiento histórico real de la cartera (igual ponderación)
- Rentabilidad anual, volatilidad anual y **ratio Sharpe**
- Curva de crecimiento acumulado
- Asignación estratégica (SAA) recomendada según perfil: gráfico de dona + barras

---

## 🗂 Estructura del proyecto

```
trading-ml-sistema/
├── trading_ml_app.py     # Aplicación principal (Streamlit)
├── requirements.txt      # Dependencias Python
└── README.md             # Este archivo
```

---

## 🛠 Stack tecnológico

| Herramienta | Uso |
|---|---|
| `streamlit` | Interfaz web interactiva |
| `yfinance` | Datos de mercado reales (Yahoo Finance) |
| `scikit-learn` | Modelos ML (Random Forest, Regresión Lineal) |
| `pandas` / `numpy` | Procesamiento de series temporales |
| `plotly` | Visualizaciones interactivas (velas, RSI, MACD, predicción) |

---

## 📋 Tickers de ejemplo por categoría

```
# Índices
^IBEX     → IBEX 35 (España)
^GSPC     → S&P 500 (EEUU)
^IXIC     → NASDAQ

# Acciones españolas (Bolsa de Madrid)
SAN.MC    → Santander
BBVA.MC   → BBVA
ITX.MC    → Inditex
TEF.MC    → Telefónica

# Acciones globales
AAPL      → Apple
MSFT      → Microsoft
NVDA      → NVIDIA
AMZN      → Amazon

# ETFs
SPY       → SPDR S&P 500 ETF
QQQ       → Invesco NASDAQ-100
AGG       → iShares Core US Aggregate Bond

# Commodities y crypto
GLD       → SPDR Gold Shares
BTC-USD   → Bitcoin
ETH-USD   → Ethereum

# Divisas
EURUSD=X  → EUR/USD
```

---

## ⚠️ Aviso legal

Este proyecto es de carácter **educativo e investigador**. No constituye asesoramiento financiero ni una recomendación de inversión. Las predicciones generadas por los modelos ML son aproximaciones estadísticas y no garantizan resultados futuros. Los datos de Yahoo Finance pueden presentar un retraso de 15-20 minutos.

Antes de tomar cualquier decisión de inversión, consulta con un asesor financiero certificado.

---

## 👤 Autor

**Gustavo Javier Criao**
Ingeniería Eléctrica → Data Science & Machine Learning
[4Geeks Academy](https://4geeks.com) · Barcelona

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/https://www.linkedin.com/in/gustavo-javier-criao-187824222/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/https://github.com/javiercriao5-creator)

---

## 🔗 Proyectos relacionados

- **Oráculo Musical** — Modelo predictivo de canciones virales con ML y NLP para discográficas y agencias de publicidad *(próximamente)*

---

*Construido con Python · scikit-learn · Streamlit · Yahoo Finance*
