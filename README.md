# 環境
- 仮想環境 
    - stock_price_prediction
        -  create -n stock_price_prediction python=3.10☓
        - 3.8でないとfbprophetがconda installできない
        - 後ほどfbprophetが古いことがわかったのでインストール不要
- ライブラリ 
    - conda install pandas=1.5 
        - pandas-datareaderは、pandasのバージョンが1系でないといけないらしい
        - pandas 2.xx系はエラーになった
    - conda install -c anaconda pandas-datareader
    - conda install -c conda-forge yfinance
    - conda install prophet
    - conda install -c conda-forge streamlit
    - conda install -c plotly plotly
    - conda install -c conda-forge fbprophet
        - エラーになる。python3.10ではエラー

- conda install streamlit でエラー
```
Traceback (most recent call last):
  File "C:\Users\seiji\.conda\envs\stock_price_prediction\Scripts\streamlit-script.py", line 6, in <module>
    from streamlit.cli import main
ModuleNotFoundError: No module named 'streamlit.cli'
```
→conda install -c conda-forge streamlitでインストールしないとエラー


- 
    - conda install holidays==0.10.5.2  

- conda install joblib

- 参考サイト
    - https://qiita.com/irisu-inwl/items/9d49a14c1c67391565f8
    - https://di-acc2.com/programming/python/20621/
