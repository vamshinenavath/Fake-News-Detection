import streamlit as st

st.set_page_config(page_title="🧠 Model Architecture", layout="wide")
st.title("🧠 Model Overview")

st.markdown("""
Welcome to the **Model Overview** page!  
This application uses a two-stage machine learning pipeline to detect and analyze fake news.
""")

# Primary Classifier Section
st.header("🔍 Primary Classifier – Real vs Fake")
st.markdown("""
The **Primary Classifier** is responsible for the **binary classification** of news articles:

- ✅ **Real News**
- 🚫 **Fake News**

It takes both the **title** and **content** of the article as input and determines whether the article is likely to be truthful or deceptive.


**Underlying model**:  
CNN + BiLSTM.
""")

# Secondary Classifier Section
st.header("🧪 Secondary Classifier – Fake News Multi-Classifier")
st.markdown("""
If an article is classified as **Fake**, it is further passed to a **Secondary Classifier** to determine its **specific category**.

###  Fake News Categories:
- 🧠 `bias`
- 🎯 `clickbait`
- 🧩 `conspiracy`
- ❌ `fake`
- 💢 `hate`
- 🧪 `junksci`
- 🏛️ `political`
- 📢 `rumor`
- 🤡 `satire`
- 🚫 `unreliable`

**Model input**: The same title and content  
**Output**: One of the above classes

**Underlying Model**:
    LSTM + FCNN
""")