import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

# 页面配置：宽屏显示
st.set_page_config(layout="wide")

# 读取数据
df = pd.read_csv("df_standardized.csv", index_col=0)
df.index = pd.to_datetime(df.index)
columns = df.columns.tolist()

# 页面标题
st.title("Time Series Visualization")

# 选择两条线
selected = st.multiselect("请选择两条时间序列进行回归分析（最多两条）", columns, default=columns[:2])

# 绘图
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
    title='Time Series',
    xaxis_title='Date',
    yaxis_title='Standardized Values',
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# 回归分析
if len(selected) == 2:
    y_col, x_col = selected[0], selected[1]
    data = df[[y_col, x_col]].dropna()
    X = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()

    st.subheader("回归结果")
    st.markdown(f"""
    **回归模型**: `{y_col} ~ {x_col}`  
    **观测数**: {int(model.nobs)}  
    **β 系数**: {model.params[1]:.4f}  
    **p 值**: {model.pvalues[1]:.4f}  
    **R²**: {model.rsquared:.4f}
    """)
elif len(selected) < 2:
    st.info("请选择两条线进行回归分析")
else:
    st.warning("一次只能选择两条线进行回归分析")
