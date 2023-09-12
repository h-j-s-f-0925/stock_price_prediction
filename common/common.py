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

from joblib import Memory

import asyncio

@st.cache_data
def load_model():
    model = Prophet(daily_seasonality=False)
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


async def get_forecast_async(model):
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast


# プログレスバーの定義
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


# キャッシュディレクトリの設定
memory = Memory("cache_directory", verbose=0)
@memory.cache
def cached_cross_validation(model, initial, period, horizon):
    return cross_validation(model, initial=initial, period=period, horizon=horizon)

# クロスバリデーションと性能指標の取得
def get_cross_validation_results(model, date_st, date_fn):
    # year_stとyear_endのスライダーから取得した情報を基に、initial、period、horizonを適切に設定する
    total_days = (date_fn - date_st).days
    # 期間の設定
    initial_days = int(total_days * 0.5)
    period_days = int(total_days * 0.25)
    horizon_days = int(total_days * 0.25)
    
    df_cv = cached_cross_validation(model, initial=f"{initial_days} days", period=f"{period_days} days", horizon=f"{horizon_days} days")
    df_p = performance_metrics(df_cv)
    df_p['horizon'] = df_p['horizon'].dt.days
    return df_cv, df_p

# メトリクスの結果をグラフにプロット
def plot_metrics(df_p, result_option):
    fig, ax = plt.subplots(figsize=(10, 6))
    if result_option == "MSE":
        ax.plot(df_p['horizon'], df_p['mse'], label='MSE')
    elif result_option == "RMSE":
        ax.plot(df_p['horizon'], df_p['rmse'], label='RMSE')
    elif result_option == "MAE":
        ax.plot(df_p['horizon'], df_p['mae'], label='MAE')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.legend()
    ax.set_title('Metrics over Time')
    ax.set_xlabel('Horizon')
    ax.set_ylabel(f'{result_option}')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# 3. 評価結果を可視化
def visualize_evaluation(df_cv, result_option):
    if result_option == "MSE":
        fig_mse = plot_cross_validation_metric(df_cv, metric='mse')
        st.pyplot(fig_mse)
    elif result_option == "RMSE":
        fig_rmse = plot_cross_validation_metric(df_cv, metric='rmse')
        st.pyplot(fig_rmse)
    elif result_option == "MAE":
        fig_mae = plot_cross_validation_metric(df_cv, metric='mae')
        st.pyplot(fig_mae)

# 4. トレンドと周期性の可視化
def plot_trends(model, df_cv):
    forecast = model.predict(df_cv)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# 主要関数: メトリクス結果の表示
def display_metrics(ticker, model, result_option, date_st, date_fn):
    st.write(f"### {ticker} のメトリクスの結果")

    # プログレスバーの初期化
    progress = ProgressBar(start_message='クロスバリデーションを開始しています...(時間がかかります)')
    progress.update(50, 100)  # 50%完了と仮定

    df_cv, df_p = get_cross_validation_results(model, date_st, date_fn)
    progress.update(75, 100)  # 75%完了と仮定

    st.write(f'### 評価結果の可視化 {result_option}')
    # plot_metrics(df_p, result_option)
    visualize_evaluation(df_cv, result_option)
    st.markdown("""
    - X軸 (horizon): これは予測が行われた未来の日数を示しています。たとえば、horizonが30 daysの場合、これは30日後の予測の精度を示しています。

    - Y軸: ここには特定の性能指標（例：RMSE, MAEなど）の値がプロットされます。値が低いほど、その指標における予測の誤差が小さいことを示します。

    - プロットのトレンドの解釈
        - 上昇トレンド: もし指標がhorizonの増加に伴って上昇する場合、それは予測の誤差が時間の経過とともに増加していることを意味します。これは、モデルが遠い未来の予測に苦労していることを示唆している可能性があります。
        - 安定したトレンド: もし指標がhorizonに対して安定しているか、変動が少ない場合、モデルは様々な期間にわたって一貫した性能を持っていると言えます。
        - 下降トレンド: このようなトレンドは一般的ではありませんが、もし見られる場合は、モデルが短期間の予測よりも長期間の予測でより正確である可能性があります。
    """)
    
    
    st.write('### トレンドと周期性の結果')
    plot_trends(model, df_cv)

    # 最後にプログレスバーを100%完了にしてメッセージを表示
    progress.done(completion_text="クロスバリデーションが完了しました！")
