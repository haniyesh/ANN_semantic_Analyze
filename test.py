import pandas as pd
df=pd.read_csv("/home/haniye/crypto_news_ann/news_cleaned.csv")
print(df["weight"].describe())
print(df["weight"].value_counts().sort_index())