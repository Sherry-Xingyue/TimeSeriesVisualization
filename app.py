import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

# 读取数据
df = pd.read_csv("df_standardized.csv", index_col=0)
df.index = pd.to_datetime(df.index)
columns = df.columns.tolist()

st.set_page_config(layout="wide")  # 响应式页面更宽

# 页面标题
st.title("交互式时间序列可视化与回归分析")

# 初始选择
selected = st.session_state.get("selected_vars", columns[:2])

# 绘图（最上方）
fig = go.Figure()
for col in columns:
    fig.add_trace(go.Scattergl(
        x=df.index,
        y=df[col],
        mode='lines',
        name=col,
        visible=True if col in selected else 'legendonly'
    ))

fig.update_layout(
    height=500,
    title='时间序列图',
    xaxis_title='日期',
    yaxis_title='标准化值',
    margin=dict(l=40, r=40, t=50, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# 变量选择（图下面）
selected_vars = st.multiselect("请选择两条时间序列进行回归", columns, default=selected)
st.session_state["selected_vars"] = selected_vars  # 保持状态同步

# 回归分析（变量选择下面）
if len(selected_vars) == 2:
    y_col, x_col = selected_vars[0], selected_vars[1]
    data = df[[y_col, x_col]].dropna()
    X = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()

    st.subheader("回归结果")
    st.markdown(f"""
**回归模型**: `{y_col} ~ {x_col}`  
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
