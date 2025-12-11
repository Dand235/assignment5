import pandas
from outliers import smirnov_grubbs as grubbs
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import base64
from io import BytesIO
from xhtml2pdf import pisa

if "refresh_key" not in st.session_state:
    st.session_state["refresh_key"] = 0


@st.cache_data
def load_and_prepare(refresh_key):
    df = pandas.read_csv(
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSuNU434Ir8wib_9YpUzMByZzD5mPE34-2uf3fJ643kQqfwyGGkyuO-8iGel2vcLW8GgqzRY71acSEo/pub?gid=0&single=true&output=csv")
    return df


def convert_html_to_pdf(source_html):
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(source_html.encode("utf-8")), result)
    if not pdf.err:
        return result.getvalue()
    return None


def fig_to_base64(fig):
    img_bytes = fig.to_image(format="png", width=800, height=500)
    return base64.b64encode(img_bytes).decode()


df = load_and_prepare(st.session_state.get("refresh_key", 0))
selected = df[13:]
selected = selected.dropna(axis=1).reset_index(drop=True)
columns = selected.iloc[0].to_list()
selected = selected.drop(0).reset_index(drop=True)
print(columns)
selected.columns = columns
selected.to_html("table.html")
only_mines = selected.iloc[:, 1:].reset_index(drop=True)

only_mines.to_html("mines.html")
only_mines = only_mines.apply(pandas.to_numeric, errors='coerce')

averages = only_mines.mean(axis=1)
dates = selected.iloc[:, 0]
dates_average = pandas.concat([dates, averages], axis=1)
dates_average.columns = ['Date', 'Average']

window_size = 10
mine_count = 0

mine_columns = only_mines.columns.to_list()
# df_int = only_mines.astype(int)
df_rounded = only_mines.round(3)
date_rounded_val = pandas.concat([dates, df_rounded], axis=1)
st.title("Task #5")
st.write(dates_average)

col1, col2 = st.columns(2)
with col1:
    z_threshold = st.slider(
        "Z-score threshold",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Higher = fewer outliers (3.0 ≈ 99.7% of data within bounds)"
    )
    k_iqr = st.slider(
        "IQR multiplier (k)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="1.5 = standard mild outliers, 3.0 = only extreme outliers"
    )

with col2:
    move_avg_pct = st.slider(
        "Moving Avg deviation (%)",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Point must deviate by this % from rolling mean"
    )
    move_avg_percent = move_avg_pct / 100.0

    alpha_grubbs = st.slider(
        "Grubbs test significance (α)",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Lower α = stricter test → fewer outliers"
    )

tabs = st.tabs(mine_columns)

dict_of_outliers = {}
for tab in tabs:

    with tab:
        mine_name = mine_columns[mine_count]
        first_column = selected.iloc[:, 0]
        mine_series = only_mines.iloc[:, mine_count]

        print(F'Mean is {mine_series.mean()}')
        print(F'Std : {mine_series.std()}')
        print(F'Median : {mine_series.median()}')
        col3, col4 = st.columns(2)

        with col3:
            st.metric(label="Mean:", value=round(mine_series.mean(), 3))
        with col4:
            st.metric(label="Standard Deviation:", value=round(mine_series.std(), 3))

        Q1 = mine_series.quantile(0.25)
        Q3 = mine_series.quantile(0.75)
        IQR = Q3 - Q1

        col5, col6 = st.columns(2)

        with col5:
            st.metric(label="Median:", value=round(mine_series.median(), 3))
        with col6:
            st.metric(label="Interquartile Range:", value=round(IQR, 3))

        iqr_outliers = mine_series[(mine_series < Q1 - k_iqr * IQR) | (mine_series > Q3 + k_iqr * IQR)]
        print(F'IQR : {IQR}')

        z_scores = (mine_series - mine_series.mean()) / mine_series.std()
        outliers_z = mine_series[np.abs(z_scores) > z_threshold]

        col7, col8 = st.columns(2)

        with col7:
            st.write("IQR outliers:")
            iqr_outliers = round(iqr_outliers, 3)
            date_iqr = pandas.merge(dates, iqr_outliers, left_index=True, right_index=True, how='inner')
            st.write(date_iqr)
        with col8:
            outliers_z = round(outliers_z, 3)
            date_z = pandas.merge(dates, outliers_z, left_index=True, right_index=True, how='inner')
            st.write("z score outliers:")
            st.write(date_z)

        moving_avg = mine_series.rolling(window=window_size).mean()
        percent_from_mov_avg = (mine_series - moving_avg) / moving_avg
        outliers_moving_avg = mine_series[np.abs(percent_from_mov_avg) > move_avg_percent]

        outlier_grubbs = pandas.concat([mine_series, grubbs.test(mine_series, alpha=alpha_grubbs)]).drop_duplicates(
            keep=False)

        col9, col10 = st.columns(2)

        with col9:
            st.write("Moving average outliers")
            outliers_moving_avg = round(outliers_moving_avg, 3)
            date_mov_avg = pandas.merge(dates, outliers_moving_avg, left_index=True, right_index=True, how='inner')
            st.write(date_mov_avg)

        with col10:
            st.write("Grubbs outliers:")
            if (outlier_grubbs.empty):
                st.write("No grubbs outliers")
            else:
                outlier_grubbs = round(outlier_grubbs, 3)
                date_grubbs = pandas.merge(dates, outlier_grubbs, left_index=True, right_index=True, how='inner')
                st.write(date_grubbs)
        all_outliers = pandas.concat([date_z, date_grubbs, date_mov_avg, date_iqr], axis=0).drop_duplicates()
        st.subheader("All outliers")
        st.write(all_outliers)
        dict_of_outliers[mine_columns[mine_count]] = all_outliers
    mine_count += 1

options = ['line', 'bar', 'stacked']
selection = st.segmented_control(
    "Chart Type", options, key=1
)

trendlines = [1, 2, 3, 4]
trend_select = int(st.segmented_control(
    "Trendline", trendlines, key=2, default=1, selection_mode="single"
))
fig = go.Figure()
x_num = np.arange(len(dates))
if selection == 'line':
    for col in mine_columns:
        fig.add_trace(go.Scatter(x=dates, y=date_rounded_val[col].values, mode='lines', name=col))
        coef = np.polyfit(x_num, date_rounded_val[col].values, trend_select)
        y_trend = np.polyval(coef, x_num)
        fig.add_trace(go.Scatter(x=dates, y=y_trend, mode='lines', name=f'{col} trend', line=dict(dash='dot')))

        if dict_of_outliers[col].empty == False:
            fig.add_trace(go.Scatter(
                x=dict_of_outliers[col].iloc[:, 0],
                y=dict_of_outliers[col].iloc[:, 1],
                mode='markers',
                name=f'{col} outliers',
            ))





elif selection == 'bar':

    for col in mine_columns:
        fig.add_trace(go.Bar(x=dates, y=date_rounded_val[col].values, name=col))
        coef = np.polyfit(x_num, date_rounded_val[col].values, trend_select)
        y_trend = np.polyval(coef, x_num)
        fig.add_trace(go.Scatter(x=dates, y=y_trend, mode='lines', name=f'{col} trend', line=dict(dash='dot')))

    fig.update_layout(barmode='group')


elif selection == 'stacked':
    total_y = date_rounded_val[mine_columns].sum(axis=1).values
    coef = np.polyfit(x_num, total_y, trend_select)
    y_trend = np.polyval(coef, x_num)
    fig.add_trace(go.Scatter(x=dates, y=y_trend, mode='lines', name='Total Trend'))
    for col in mine_columns:
        fig.add_trace(go.Bar(x=dates, y=date_rounded_val[col].values, name=col))
    fig.update_layout(barmode='stack')

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)



col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("Refresh data"):
        st.session_state["refresh_key"] += 1
        st.cache_data.clear()
        st.rerun()

with col_btn2:

    try:
        graph_base64 = fig_to_base64(fig)
        graph_html = f'<img src="data:image/png;base64,{graph_base64}" style="width:100%; max-width:800px;"/>'
    except:
        graph_html = "<p>Graph could not be embedded</p>"

    html_content = f"""
    <html>
    <head>
        <title>Task #5 Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Task #5 - Mining Data Report</h1>
        <h2>Chart</h2>
        {graph_html}
        <h2>Date Averages</h2>
        {dates_average.to_html(index=False)}
        <h2>Statistics Summary</h2>
        {''.join([f"<h3>{col}</h3><p><b>Mean:</b> {only_mines[col].mean():.3f} | <b>Std Dev:</b> {only_mines[col].std():.3f} | <b>Median:</b> {only_mines[col].median():.3f} | <b>Outliers:</b> {len(dict_of_outliers[col])}</p>" for col in mine_columns])}
    </body>
    </html>
    """

    pdf_bytes = convert_html_to_pdf(html_content)

    if pdf_bytes:
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="mining_report.pdf",
            mime="application/pdf"
        )
    else:

        st.error("PDF generation failed")
