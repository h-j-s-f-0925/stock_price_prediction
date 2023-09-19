import asyncio
from datetime import datetime

import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from common.common import (
    ProgressBar,
    display_metrics,
    get_data_for_ticker,
    get_data_for_tickers,
    get_forecast_async,
    load_model,
)
from prophet.plot import plot_plotly

plt.rcParams["font.family"] = "Arial"

yf.pdr_override()


# メイン処理
def main():
    st.title("株価予測アプリ")

    # Tickerの動的入力
    tickers = st.multiselect(
        "株価の推移をグラフ化したいTickerを選択してください：",
        ["VOO", "AGG", "VEA", "AAPL", "MSFT", "GOOG"],
        default=["VOO"],
        key="unique_key_1",
    )

    year_st = st.slider("開始年を選んでください。", 2010, 2023, 2010)
    date_st = datetime(year_st, 1, 1)
    year_end = st.slider("終了年を選んでください。", year_st + 2, 2023, year_st + 2)
    date_fn = datetime(year_end, 4, 1)

    # 株価データの取得
    df = get_data_for_tickers(tickers, date_st, date_fn)
    # 株価データの可視化
    st.line_chart(df["Close"])

    ticker = st.selectbox(
        "株価を予測するTickerを選択してください：",
        ["VOO", "VEA", "AGG", "AAPL", "MSFT", "GOOG"],
        #  default=["VOO"],
        key="unique_key_2",
    )

    # 表示するグラフの選択
    result_option = st.radio("表示する評価結果グラフを選択してください", ("MSE", "RMSE", "MAE"))

    if st.button("予測を開始"):
        # プログレスバーの初期化
        progress = ProgressBar(start_message="処理を開始しています。....")
        # データ取得
        df = get_data_for_ticker(ticker, date_st, date_fn)
        data = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
        # data_sampled = data.sample(frac=0.5)

        # モデルの学習
        model = load_model()
        # model.fit(data_sampled)
        model.fit(data)
        # モデル学習完了を示すプログレス更新
        progress.update(30, 100)  # 50%完了と仮定
        progress.update_message("モデルの学習が完了しました。...")

        # 予測
        forecast = asyncio.run(get_forecast_async(model))
        progress.update(50, 100)  # 50%完了と仮定
        progress.update_message("モデルの予測が完了しました。...")

        # 予測結果の可視化
        st.write(f"### {ticker} の予測結果")
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)
        # モデル学習完了を示すプログレス更新
        progress.update(70, 100)  # 50%完了と仮定
        progress.update_message("予測結果の可視化が完了しました。...")

        # メトリクスの可視化
        display_metrics(ticker, model, result_option, date_st, date_fn)
        # 最後にプログレスバーを100%完了にしてメッセージを表示
        progress.done(completion_text="処理が完了しました！")


if __name__ == "__main__":
    main()
