from itertools import combinations

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

# 自动对所有被选择的变量两两组合进行回归
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
- **斜率系数 (β₁)**: {model.params[1]:.4f}  
- **p 值 (β₁)**: {model.pvalues[1]:.4f}  
- **R²**: {model.rsquared:.4f}
        """)
