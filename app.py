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

# 标题
st.title("交互式时间序列可视化与回归分析")

# ========== 绘图设置 ==========
st.subheader("可视化变量设置")

displayed_vars = st.multiselect(
    "请选择要显示在图中的时间序列", columns, default=columns[:2])

# Shift 设置（用于图）
shift_settings_plot = {}
if displayed_vars:
    st.markdown("### 图中变量的 Shift 设置（单位：月）")
    for var in displayed_vars:
        shift_val = st.number_input(
            f"{var} 的 Shift（月）", min_value=-12, max_value=12, value=0, step=1, key=f"shift_{var}_plot")
        shift_settings_plot[var] = shift_val

# 绘图
fig = go.Figure()
for var in displayed_vars:
    shifted_series = df[var].shift(shift_settings_plot[var])
    fig.add_trace(go.Scattergl(
        x=shifted_series.index,
        y=shifted_series,
        mode='lines',
        name=f"{var} (shift={shift_settings_plot[var]})"
    ))

fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=30, b=30),
    title='时间序列图',
    xaxis_title='日期',
    yaxis_title='标准化值',
)
st.plotly_chart(fig, use_container_width=True)

# ========== 回归分析设置 ==========
st.markdown("---")
st.subheader("回归分析设置")

# 为每个变量设置 Shift（用于回归分析）
shift_settings_reg = {}
if displayed_vars:
    st.markdown("### 回归变量的 Shift 设置（单位：月）")
    for var in displayed_vars:
        shift_val = st.number_input(
            f"{var} 的 Shift（月）", min_value=-12, max_value=12, value=0, step=1, key=f"shift_{var}_reg")
        shift_settings_reg[var] = shift_val

# ========== 回归分析结果展示 ==========
if len(displayed_vars) < 2:
    st.info("请选择至少两条线进行回归分析")
else:
    st.markdown("### 回归分析结果（两两组合）")
    for y_col, x_col in combinations(displayed_vars, 2):
        y_shift, x_shift = shift_settings_reg[y_col], shift_settings_reg[x_col]

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
- **观测数**: {int(model.nobs)}  
- **截距项 (β₀)**: {model.params[0]:.4f}  
- **斜率系数 (β₁)**: {model.params[1]:.

