import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import combinations

# 页面配置
st.set_page_config(layout="wide")

# 读取数据
#df = pd.read_csv("df_standardized_finance.csv", index_col=0)
#df = pd.read_csv("df_standardized+cross_industry_dispersion.csv", index_col=0)
df = pd.read_csv("topic_share_by_gene_exp.csv", index_col=0)
df.index = pd.to_datetime(df.index)
columns = df.columns.tolist()
time_unit = 'month'

# 页面标题
st.title("Time Series Visualization")

# ========= 选择变量 =========
st.subheader("Select Variables")
displayed_vars = st.multiselect("Select time series to display in the plot", columns, default=columns[:2])

# ========= Shift 设置（同步用于图和回归）=========
shift_settings = {}
if displayed_vars:
    st.markdown("### Shift")
    for var in displayed_vars:
        shift_val = st.number_input(
            f"{var} shift (in {time_unit}s)", min_value=-12, max_value=12, value=0, step=1, key=f"shift_{var}")
        shift_settings[var] = shift_val

# ========= 绘图 =========
fig = go.Figure()
for var in displayed_vars:
    shifted_series = df[var].shift(shift_settings[var])
    fig.add_trace(go.Scattergl(
        x=shifted_series.index,
        y=shifted_series,
        mode='lines',
        name=f"{var} (shift={shift_settings[var]})"
    ))

# 设置图表样式
fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=30, b=30),
    title='Time Series',
    xaxis_title='Date',
    yaxis_title='Standardized Values',
)

# 设置 X 轴刻度显示每年
fig.update_xaxes(
    dtick="M12",
    tickformat="%Y",
    tickangle=0
)

# 设置鼠标悬停时显示年月
fig.update_traces(hovertemplate='%{x|%Y-%m}<br>%{y:.4f}')

# 展示图
st.plotly_chart(fig, use_container_width=True)

# ========= 回归分析 =========
st.markdown("---")

# shift 显示辅助函数
def shift_label(shift):
    if shift == 0:
        return "t"
    elif shift > 0:
        return f"t-{shift}"
    else:
        return f"t+{abs(shift)}"

if len(displayed_vars) < 2:
    st.info("Please select at least two series for regression analysis.")
else:
    st.subheader("Regression Results")
    st.markdown("###### OLS, no winsor")
    for y_col, x_col in combinations(displayed_vars, 2):
        y_shift, x_shift = shift_settings[y_col], shift_settings[x_col]

        # Shift 后的数据
        y_data = df[y_col].shift(y_shift)
        x_data = df[x_col].shift(x_shift)

        data = pd.concat([y_data, x_data], axis=1).dropna()
        data.columns = ['Y', 'X']

        if len(data) < 10:
            st.markdown(f"#### `{y_col} ~ {x_col}`")
            st.warning("Not enough data points for valid regression.")
            continue

        X = sm.add_constant(data['X'])
        model = sm.OLS(data['Y'], X).fit()

        st.markdown(f"#### `{y_col}({shift_label(y_shift)}) ~ {x_col}({shift_label(x_shift)})`")
        st.markdown(f"""
| Slope (β₁) | p-value (β₁) | Intercept (β₀) | R² | Observations |
|------------|--------------|----------------|-----|--------------|
| {model.params[1]:.4f} | {model.pvalues[1]:.4f} | {model.params[0]:.4f} | {model.rsquared:.4f} | {int(model.nobs)} |
""")
