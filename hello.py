import streamlit as st

st.write("Hello, Streamlit!")
st.text_input("Enter your name:")

import pandas as pd

data = pd.read_csv("datasets/news_sample.csv")
st.write(data)