import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Helpers
# -----------------------------
MONTHS = 12
CLIMATE_COLS = ["PRECIP_TOTAL_MM", "RH2M_PCT", "TEMP_MEAN_C", "TEMP_MAX_C", "TEMP_MIN_C"]

def month_sin_cos(month_num: int):
    msin = np.sin(2 * np.pi * month_num / 12)
    mcos = np.cos(2 * np.pi * month_num / 12)
    return msin, mcos

def build_supervised(df, lookback=12):
    """
    Builds a supervised dataset:
    X = flattened climate of previous 12 months + month sin/cos of target month
    y = dengue cases of target month
    Uses ONLY rows where DENGUE_CASES is not NaN.
    """
    df = df.sort_values("DATE").reset_index(drop=True)

    X, y, dates = [], [], []

    for i in range(lookback, len(df)):
        if pd.isna(df.loc[i, "DENGUE_CASES"]):
            continue

        # Ensure consecutive monthly steps (important for time-series)
        dates_window = df.loc[i-lookback:i, "DATE"].tolist()
        ok = True
        for j in range(1, len(dates_window)):
            exp = dates_window[j-1] + pd.offsets.MonthBegin(1)
            if dates_window[j] != exp:
                ok = False
                break
        if not ok:
            continue

        # Past 12 months climate
        seq = df.loc[i-lookback:i-1, CLIMATE_COLS].to_numpy(dtype=float)  # (12, 5)
        seq_flat = seq.reshape(-1)  # (60,)

        # Seasonality for target month
        m = int(df.loc[i, "DATE"].month)
        msin, mcos = month_sin_cos(m)

        row = np.concatenate([seq_flat, [msin, mcos]])
        X.append(row)
        y.append(float(df.loc[i, "DENGUE_CASES"]))
        dates.append(df.loc[i, "DATE"])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    dates = pd.to_datetime(dates)

    return X, y, dates

def train_test_split_time(X, y, dates, train_ratio=0.8):
    n = len(dates)
    split = int(train_ratio * n)
    return (X[:split], y[:split], dates[:split],
            X[split:], y[split:], dates[split:])

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Nueva Vizcaya Dengue AI Dashboard", layout="wide")
st.title("Nueva Vizcaya Dengue AI Dashboard (Climate + Dengue Forecasting)")

st.write("This dashboard loads your monthly dataset (NASA POWER climate + dengue cases) and produces forecasts using an ML pipeline aligned with your Chapter 3 method.")

# Load data
DEFAULT_PATH = "data/Nueva_Vizcaya_Climate_Dengue_LSTM_Ready_UPDATED_VALUES.xlsx"

with st.sidebar:
    st.header("Data Source")
    data_path = st.text_input("Excel path in repo", value=DEFAULT_PATH)
    sheet_name = st.text_input("Sheet name", value="Integrated_Monthly")

    st.header("Model Settings")
    train_ratio = st.slider("Train ratio (time-ordered)", 0.6, 0.9, 0.8, 0.05)

    st.caption("Baseline model runs automatically online. LSTM mode can be added later if you upload a trained model file.")

# Read Excel
try:
    df = pd.read_excel(data_path, sheet_name=sheet_name)
except Exception as e:
    st.error(f"Could not read Excel file/sheet. Error: {e}")
    st.stop()

# Standardize expected columns
# Expected minimum: DATE + climate columns + DENGUE_CASES
# If your sheet uses different names, rename here.
rename_map = {}
for c in df.columns:
    if c.upper() == "DATE":
        rename_map[c] = "DATE"
df = df.rename(columns=rename_map)

# Try to auto-detect dengue column
if "DENGUE_CASES" not in df.columns:
    # common variants
    for cand in ["DENGUE", "CASES", "DENGUE_CASE", "DENGUE_CASES_TARGET"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "DENGUE_CASES"})
            break

# Ensure DATE is datetime
df["DATE"] = pd.to_datetime(df["DATE"])

missing_cols = [c for c in ["DATE", "DENGUE_CASES"] + CLIMATE_COLS if c not in df.columns]
if missing_cols:
    st.error("Missing required columns in the selected sheet:\n\n" + "\n".join(missing_cols))
    st.stop()

# Show data preview
with st.expander("Preview dataset"):
    st.dataframe(df.head(24))

# Plot dengue over time
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dengue Cases Over Time")
    fig = plt.figure()
    plt.plot(df["DATE"], df["DENGUE_CASES"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Dengue cases")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("Latest Values")
    latest = df.dropna(subset=["DENGUE_CASES"]).sort_values("DATE").iloc[-1]
    st.metric("Latest month", latest["DATE"].strftime("%Y-%m"))
    st.metric("Latest dengue cases", int(latest["DENGUE_CASES"]))

# Climate plots
st.subheader("Climate Variables (NASA POWER)")
fig2 = plt.figure()
for c in CLIMATE_COLS:
    plt.plot(df["DATE"], df[c], label=c)
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)

# Build supervised dataset
X, y, dates = build_supervised(df, lookback=MONTHS)

if len(y) < 20:
    st.warning("Not enough valid rows to train/test. Make sure dengue cases exist for enough months.")
    st.stop()

X_train, y_train, d_train, X_test, y_test, d_test = train_test_split_time(X, y, dates, train_ratio=train_ratio)

# Train a fast baseline model (good for web hosting)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

st.subheader("Model Evaluation (Baseline ML)")
st.write(f"**MAE:** {mae:.2f} cases")
st.write(f"**RMSE:** {rmse:.2f} cases")

# Plot predicted vs actual on test set
st.subheader("Actual vs Predicted (Test period)")
fig3 = plt.figure()
plt.plot(d_test, y_test, marker="o", label="Actual")
plt.plot(d_test, pred_test, marker="o", label="Predicted")
plt.xlabel("Date")
plt.ylabel("Dengue cases")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig3)

# Predict next month
st.subheader("Predict Next Month")
last_date = df["DATE"].max()
next_date = (last_date + pd.offsets.MonthBegin(1))
st.write(f"Next month to predict: **{next_date.strftime('%Y-%m')}**")

# Build next-month feature vector from the last 12 months climate
df_sorted = df.sort_values("DATE").reset_index(drop=True)
last_window = df_sorted.iloc[-MONTHS:]  # last 12 rows
if len(last_window) < MONTHS:
    st.warning("Not enough months in dataset for a 12-month lookback.")
else:
    seq = last_window[CLIMATE_COLS].to_numpy(dtype=float).reshape(-1)
    msin, mcos = month_sin_cos(int(next_date.month))
    x_next = np.concatenate([seq, [msin, mcos]]).reshape(1, -1)

    pred_next = float(model.predict(x_next)[0])
    st.metric("Predicted dengue cases (next month)", f"{pred_next:.0f}")

# Download predictions
out = pd.DataFrame({"DATE": d_test, "ACTUAL": y_test, "PREDICTED": pred_test})
csv = out.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Test Predictions (CSV)",
    data=csv,
    file_name="test_predictions.csv",
    mime="text/csv"
)

