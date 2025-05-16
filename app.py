import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

# 配置页面
st.set_page_config(layout="wide")

# 数据读取
df = pd.read_csv("df_standardized.csv", index_col=0)
df.index = pd.to_datetime(df.index)
columns = df.columns.tolist()

# 页面标题
st.title("交互式时间序列可视化与回归分析")

# ====== 图表展示区域 ======
# 选择要显示在图中的变量（可以多选）
displayed_vars = st.multiselect("请选择要显示在图中的时间序列", columns, default=columns[:2])

# 为每个显示的变量提供 shift 选项（图用）
shift_settings_display = {}
st.markdown("### 图像变量的 Shift 设置（单位：月）")
for var in displayed_vars:
    shift_val = st.number_input(f"{var} 的 Shift（月）", min_value=-12, max_value=12, value=0, step=1, key=f"shift_{var}_display")
    shift_settings_display[var] = shift_val

# 绘图
fig = go.Figure()
for col in displayed_vars:
    shifted_series = df[col].shift(shift_settings_display[col])
    fig.add_trace(go.Scattergl(
        x=df.index,
        y=shifted_series,
        mode='lines',
        name=f"{col} (Shift={shift_settings_display[col]})",
        visible=True
    ))

fig.update_layout(
    height=500,
    title="时间序列图（可调 Shift）",
    xaxis_title='日期',
    yaxis_title='标准化值',
    margin=dict(l=40, r=40, t=50, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ====== 回归分析区域 ======
st.markdown("---")
st.subheader("回归分析设置")

# 选择两个变量进行回归
selected_vars = st.multiselect("请选择两条时间序列进行回归", columns, default=columns[:2], key="regression_selection")

# 为回归变量设置 Shift
shift_settings_reg = {}
if len(selected_vars) >= 1:
    st.markdown("### 回归变量的 Shift 设置（单位：月）")
    for var in selected_vars:
        shift_val = st.number_input(f"{var} 的 Shift（月）", min_value=-12, max_value=12, value=0, step=1, key=f"shift_{var}_reg")
        shift_settings_reg[var] = shift_val

# 回归分析逻辑
if len(selected_vars) == 2:
    y_col, x_col = selected_vars[0], selected_vars[1]
    y_shift, x_shift = shift_settings_reg[y_col], shift_settings_reg[x_col]

    # Shift 后对齐数据
    y_data = df[y_col].shift(y_shift)
    x_data = df[x_col].shift(x_shift)

    data = pd.concat([y_data, x_data], axis=1).dropna()
    data.columns = ['Y', 'X']
    X = sm.add_constant(data['X'])
    model = sm.OLS(data['Y'], X).fit()

    st.markdown("### 回归结果")
    st.markdown(f"""
**回归模型**: `{y_col}(t{f"+{y_shift}" if y_shift > 0 else f"{y_shift}" if y_shift < 0 else ""}) ~ {x_col}(t{f"+{x_shift}" if x_shift > 0 else f"{x_shift}" if x_shift < 0 else ""})`  
**观测数**: {int(model.nobs)}  
**截距项 (β₀)**: {model.params[0]:.4f}  
**斜率系数 (β₁)**: {model.params[1]:.4f}  
**p 值 (β₁)**: {model.pvalues[1]:.4f}  
**R²**: {model.rsquared:.4f}
    """)

elif len(selected_vars) < 2:
    st.info("请选择两条线进行回归分析")
else:
    st.warning("一次只能选择两条线进行回归分析")
