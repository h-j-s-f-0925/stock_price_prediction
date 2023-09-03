import pandas as pd
import time
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
def loadmodel():
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

class ProgressBar:
    def __init__(self, start_message='処理を開始しています...'):
        self.progress_placeholder = st.empty()
        self.progress_bar = self.progress_placeholder.progress(0)
        self.text_placeholder = st.empty()
        self.text_placeholder.text(start_message)

    def update(self, value, max_value):
        progress_percent = int((value / max_value) * 100)
        self.progress_bar.progress(progress_percent)

    def update_message(self, message):
        self.text_placeholder.text(message)

    def done(self, completion_text="処理が完了しました！"):
        self.progress_bar.progress(100)
        self.text_placeholder.text(completion_text)
        time.sleep(2)  # 2秒間メッセージを表示してから消す
        self.progress_placeholder.empty()
        self.text_placeholder.empty()

# ------------------------------------------------------------------------------------------

st.title("株価予測アプリ")

year_st = st.slider('開始年を選んでください。', 2010, 2023, 2010)
date_st = datetime(year_st, 1, 1)
year_end = st.slider("終了年を選んでください。", year_st + 1, 2023, year_st + 1)
date_fn = datetime(year_end, 4, 1)

# Tickerの動的入力
tickers = st.multiselect("株価の推移をグラフ化したいTickerを選択してください：",
                         ["VOO", "AGG", "VEA", "AAPL", "MSFT", "GOOG"], 
                         default=["VOO"],
                         key="unique_key_1")
df = get_data_for_tickers(tickers, date_st, date_fn)

st.line_chart(df["Close"])


import pandas as pd
from joblib import Memory

# キャッシュディレクトリの設定
memory = Memory("cache_directory", verbose=0)

@memory.cache
def cached_cross_validation(model, initial, period, horizon):
    return cross_validation(model, initial=initial, period=period, horizon=horizon)


def display_metrics(ticker, model):
    st.write(f"### {ticker} のメトリクスの結果")
    
    # # メトリクスの説明
    # - MSE（平均二乗誤差）
    # - RMSE（平均平方二乗誤差）
    # - MAE（平均絶対誤差）
    # - MAPE（平均絶対パーセント誤差）
    # - COVERAGE（予測値の上限yhat_lowerから予測値の下限yhat_upperの範囲）

    # プログレスバーの初期化
    progress = ProgressBar(start_message='クロスバリデーションを開始しています...(時間がかかります)')
    progress.update(50, 100)  # 50%完了と仮定

    # クロスバリデーションと性能指標の取得
    # キャッシュを使用してクロスバリデーションを実行
    df_cv = cached_cross_validation(model, initial='730 days', period='360 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    df_p['horizon'] = df_p['horizon'].dt.days
    # パフォーマンスメトリクス完了を示すプログレス更新
    progress.update(75, 100)  # 75%完了と仮定

    option = st.radio(
        "表示するグラフを選択してください",
        ("MSE", "RMSE", "MAE")
    )

    if option == "MSE":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_p['horizon'], df_p['mse'], label='MSE')
    elif option == "RMSE":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_p['horizon'], df_p['rmse'], label='RMSE')
    elif option == "MAE":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_p['horizon'], df_p['mae'], label='MAE')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.legend()
    ax.set_title('Metrics over Time')
    ax.set_xlabel('Horizon')
    ax.set_ylabel(f'{option}')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # MSEの評価結果の可視化
    st.write("### 評価結果を可視化")
    st.caption(f"横軸はHorizon（予測値の範囲）であり、縦軸と青線が評価指標（今回の場合{option}）を示しています。")
    if option == "MSE":
        fig_mse = plot_cross_validation_metric(df_cv, metric='mse')
        st.pyplot(fig_mse)
    elif option == "RMSE":
        fig_rmse = plot_cross_validation_metric(df_cv, metric='rmse')
        st.pyplot(fig_rmse)
    elif option == "MAE":
        fig_mae = plot_cross_validation_metric(df_cv, metric='mae')
        st.pyplot(fig_mae)

    # 予測結果の可視化
    st.write('### トレンド性と周期性')
    forecast = model.predict(df_cv)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
    
    # 最後にプログレスバーを100%完了にしてメッセージを表示
    progress.done(completion_text="クロスバリデーションが完了しました！")

ticker = st.selectbox("株価を予測するTickerを選択してください：",
                         ["VOO", "VEA", "AGG", "AAPL", "MSFT", "GOOG"],
                        #  default=["VOO"],
                         key="unique_key_2")


# プログレスバーの初期化
progress = ProgressBar(start_message="処理を開始しています。....")
# データ取得
df = get_data_for_ticker(ticker, date_st, date_fn)
data = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# モデルの学習
model = loadmodel()
model.fit(data)
# モデル学習完了を示すプログレス更新
progress.update(50, 100)  # 50%完了と仮定
progress.update_message("モデルの学習が完了しました。...")

# 予測
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 予測結果の可視化
st.write(f"### {ticker} の予測結果")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig, use_container_width=True)
# モデル学習完了を示すプログレス更新
progress.update(70, 100)  # 50%完了と仮定
progress.update_message("予測結果の可視化が完了しました。...")

# メトリクスの可視化
display_metrics(ticker, model)
# 最後にプログレスバーを100%完了にしてメッセージを表示
progress.done(completion_text="処理が完了しました！")