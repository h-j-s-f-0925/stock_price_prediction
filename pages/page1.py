import pandas as pd
from datetime import datetime
import yfinance as yf
import pandas_datareader.data as pdr

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics

import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.rcParams["font.family"] = "Arial"

yf.pdr_override()

@st.cache_data
def load_model():
    # 学習済みモデルをロードするコードをここに追加
    # ダミーの場合
    model = Prophet()
    return model

@st.cache_data
def get_data_for_ticker(ticker, start, end):
    df = pdr.get_data_yahoo(ticker, start=start, end=end)
    return df

@st.cache_data
def get_data_for_tickers(tickers, start, end):
    dfs = [pdr.get_data_yahoo(ticker, start=start, end=end) for ticker in tickers]
    # keys引数で各データフレームにラベルを付ける
    df2 = pd.concat(dfs, axis=0, keys=tickers)
    # 最も外側のインデックス（レベル0）をアンスタック
    df2 = pd.concat(dfs, axis=0, keys=tickers).unstack(0)
    return df2

date_st = datetime(2010, 1, 1)
date_fn = datetime(2023, 4, 1)


st.title("株価予測アプリ")

# Tickerの動的入力
tickers = st.multiselect("Tickerを選択してください：",
                         ["VOO", "AAPL", "MSFT", "GOOG"], 
                         default=["VOO"],
                         key="unique_key_1")
df = get_data_for_tickers(tickers, date_st, date_fn)

st.line_chart(df["Close"])

tickers = st.multiselect("Tickerを選択してください：",
                         ["VOO", "AAPL", "MSFT", "GOOG"],
                         default=["VOO"],
                         key="unique_key_2")

for ticker in tickers:
    # データ取得
    df = get_data_for_ticker(ticker, date_st, date_fn)
    data = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # モデルの学習
    model = load_model()
    model.fit(data)
    
    # 予測
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    # 予測結果の可視化
    st.write(f"### {ticker} の予測結果")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)
    
    # メトリクスの取得と可視化
    st.write(f"### {ticker} のメトリクスの結果")
    """
    - MSE（平均二乗誤差）
    - RMSE（平均平方二乗誤差）
        - RMSEで算出される値がそのまま予測値の単位として利用できるのが特徴的です。例えば、モデルの予測値の単位が金額[円]であれば、RMSEの単位も金額[円]として扱うことができる
    - MAE（平均絶対誤差）
        - MAEはRMSEと比較して外れ値に強いという特徴があります。そのため、データセットに外れ値が多く含まれる場合に有効な指標言えます。
        - RMSEと同様に得られる値がそのまま予測値の単位
    - MAPE（平均絶対パーセント誤差）
    - COVERAGE（予測値の上限yhat_lowerから予測値の下限yhat_upperの範囲）
    """
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    # df_p['horizon'] = df_p['horizon'].astype(str) # streamlitで、表示されないので、timedelta型→str型
    df_p['horizon'] = df_p['horizon'].dt.days # streamlitで、表示されないので、timedelta型→str型
    df_p
    
    """
    X軸 (horizon): これは予測が行われた未来の日数を示しています。たとえば、horizonが30 daysの場合、これは30日後の予測の精度を示しています。

    Y軸: ここには特定の性能指標（例：RMSE, MAEなど）の値がプロットされます。値が低いほど、その指標における予測の誤差が小さいことを示します。

    ## プロットのトレンドの解釈
    - 上昇トレンド: もし指標がhorizonの増加に伴って上昇する場合、それは予測の誤差が時間の経過とともに増加していることを意味します。これは、モデルが遠い未来の予測に苦労していることを示唆している可能性があります。
    - 安定したトレンド: もし指標がhorizonに対して安定しているか、変動が少ない場合、モデルは様々な期間にわたって一貫した性能を持っていると言えます。
    - 下降トレンド: このようなトレンドは一般的ではありませんが、もし見られる場合は、モデルが短期間の予測よりも長期間の予測でより正確である可能性があります。
    """
    # fig, ax = plt.subplots(figsize=(10,6))
    # ax.plot(df_p['horizon'], df_p['mse'], label='MSE')
    # ax.plot(df_p['horizon'], df_p['rmse'], label='RMSE')
    # ax.plot(df_p['horizon'], df_p['mae'], label='MAE')
    # # x軸の目盛の間隔を調整
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # 1ヶ月ごとに目盛りを表示
    # # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
    # ax.legend()
    # ax.set_title('Metrics over Time')
    # ax.set_xlabel('Horizon')
    # ax.set_ylabel('Value')
    # ax.tick_params(axis='x', rotation=45)
    # st.pyplot(fig)
    
    # ラジオボタンで表示するグラフを選択
option = st.radio(
    "表示するグラフを選択してください",
    ("MSE", "RMSE", "MAE")
)

if option == "MSE":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_p['horizon'], df_p['mse'], label='MSE')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1ヶ月ごとに目盛りを表示
    ax.legend()
    ax.set_title('Metrics over Time')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

elif option == "RMSE":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_p['horizon'], df_p['rmse'], label='RMSE')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1ヶ月ごとに目盛りを表示
    ax.legend()
    ax.set_title('Metrics over Time')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

elif option == "MAE":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_p['horizon'], df_p['mae'], label='MAE')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1ヶ月ごとに目盛りを表示
    ax.legend()
    ax.set_title('Metrics over Time')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    st.write("### 評価結果を可視化(mse)")
    st.caption("横軸はHorizon（予測値の範囲）であり、縦軸と青線が評価指標（今回の場合mse）を示しています。")
    fig_mse = plot_cross_validation_metric(df_cv, metric='mse')
    st.pyplot(fig_mse)

    st.write('### モデルの予測結果と実測値')
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)