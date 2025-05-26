import pandas as pd
import streamlit as st

def detect_outliers(df, column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    return (df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)

def check_duplicates(df):
    dups = df.duplicated().sum()
    st.info(f"🔁 Дубликатов: {dups}")
    return df.drop_duplicates()
