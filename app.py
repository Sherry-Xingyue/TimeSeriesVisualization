import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import combinations

# 页面配置
st.set_page_config(layout="wide")

# 读取数据
df = pd.read_csv("df_standardized.csv", index_col=0)
df.index = pd.to_datetime(df.index)
columns = df.columns.tolist()
time_unit = 'month'

# 页面标题
st.title("Time Series Visualization")

# ========== 选择变量 ==========
st.subheader("Slecte Variables")
displayed_vars = st.multiselect("请选择要显示在图中的时间序列", columns, default=columns[:2])

# ========== Shift 设置（用于图和回归）==========
shift_settings = {}
if displayed_vars:
    st.markdown("### Shift")
    for var in displayed_vars:
        shift_val = st.number_input(
            f"{var} shift (in {time_unit}s)", min_value=-12, max_value=12, value=0, step=1, key=f"shift_{var}")
        shift_settings[var] = shift_val

# ========== 绘图 ==========
fig = go.Figure()
for var in displayed_vars:
    shifted_series = df[var].shift(shift_settings[var])
    fig.add_trace(go.Scattergl(
        x=shifted_series.index,
        y=shifted_series,
        mode='lines',
        name=f"{var} (shift={shift_settings[var]})"
    ))

fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=30, b=30),
    title='Time Series',
    xaxis_title='Date',
    yaxis_title='Standardized Values',
)
st.plotly_chart(fig, use_container_width=True)

# ========== 回归分析 ==========
st.markdown("---")
st.subheader("regression(OLS, no winsor)")

if len(displayed_vars) < 2:
    st.info("请选择至少两条线进行回归分析")
else:
    st.markdown("### regression results")
    for y_col, x_col in combinations(displayed_vars, 2):
        y_shift, x_shift = shift_settings[y_col], shift_settings[x_col]

        # Shift 后的数据
        y_data = df[y_col].shift(y_shift)
        x_data = df[x_col].shift(x_shift)

        data = pd.concat([y_data, x_data], axis=1).dropna()
        data.columns = ['Y', 'X']

        if len(data) < 10:
            st.markdown(f"#### `{y_col} ~ {x_col}`")
            st.warning("数据点太少，无法进行有效回归")
            continue

        X = sm.add_constant(data['X'])
        model = sm.OLS(data['Y'], X).fit()

        st.markdown(f"#### `{y_col}(t{f'+{y_shift}' if y_shift > 0 else y_shift if y_shift < 0 else ''}) ~ {x_col}(t{f'+{x_shift}' if x_shift > 0 else x_shift if x_shift < 0 else ''})`")
        st.markdown(f"""
| Observations | Intercept (β₀) | Slope (β₁) | p-value (β₁) | R² |
|--------|--------------|----------------|-----------|-----|
| {int(model.nobs)} | {model.params[0]:.4f} | {model.params[1]:.4f} | {model.pvalues[1]:.4f} | {model.rsquared:.4f} |
""")

