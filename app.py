import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
import statsmodels.api as sm
import pandas as pd
import os

# 使用相对路径读取数据
data_path = os.path.join("data", "df_standardized.csv")
df = pd.read_csv(data_path,index_col=0)
plot_columns = df.columns.tolist()

# 初始化图
def create_figure(visible_lines=None):
    fig = go.Figure()
    for col in plot_columns:
        fig.add_trace(go.Scattergl(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            visible=True if not visible_lines or col in visible_lines else 'legendonly'
        ))

    fig.update_layout(
        title='时间序列图（请选择两条线以进行回归）',
        xaxis=dict(title='Date', tickformat='%Y-%m', dtick="M12", tickangle=45),
        yaxis_title='Standardized Value',
        height=800,
        width=2400
    )
    return fig

# 初始化 Dash App
app = Dash(__name__)
app.layout = html.Div([
    html.H2("交互式回归分析工具"),
    dcc.Graph(id='ts-plot', figure=create_figure()),
    html.Div(id='regression-output', style={'whiteSpace': 'pre-line', 'fontSize': '18px'})
])

# 监听图中哪些线是“visible”
@app.callback(
    Output('regression-output', 'children'),
    Input('ts-plot', 'restyleData')
)
def update_regression(restyle_data):
    if restyle_data is None:
        return "👆 请只选中任意两条线，即可查看它们的回归关系"

    # 获取当前哪些 trace 是 visible 的
    if 'visible' not in restyle_data[0]:
        return ""

    # 计算当前 visible 的 trace 索引
    visible_vals = restyle_data[0]['visible']
    changed_idxs = restyle_data[1]

    # 记录哪些线是可见的
    visible_states = [True] * len(plot_columns)
    for i, idx in enumerate(changed_idxs):
        state = visible_vals[i]
        visible_states[idx] = state != 'legendonly'

    # 找出当前显示的两条线
    selected_cols = [col for col, vis in zip(plot_columns, visible_states) if vis]
    if len(selected_cols) != 2:
        return f"❗ 当前选中了 {len(selected_cols)} 条线，请只选中 2 条线以进行回归分析"

    y_col, x_col = selected_cols[0], selected_cols[1]
    data = df[[y_col, x_col]].dropna()

    X = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()

    result = (
        f"✅ 当前回归：{y_col} ~ {x_col}\n\n"
        f"📊 观测数: {int(model.nobs)}\n"
        f"β 系数: {model.params[1]:.4f}\n"
        f"p 值: {model.pvalues[1]:.4f}\n"
        f"R²: {model.rsquared:.4f}"
    )
    return result

# 运行 App
if __name__ == '__main__':
    app.run_server(debug=True)
