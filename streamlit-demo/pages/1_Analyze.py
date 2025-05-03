import streamlit as st
from model import determine_news_validity, determine_news_category

st.title("🧪 News Analyzer")

# Load from session state if available
default_title = st.session_state.get("selected_title", "")
default_content = st.session_state.get("selected_content", "")

title = st.text_input("Enter the title of the news article:", value=default_title, key="analyzer_title")
content = st.text_area("Enter the content of the news article:", height=300, value=default_content, key="analyzer_content")

if st.button("🚀 Analyze"):
    with st.spinner("Detecting..."):
        validity = determine_news_validity(title, content)
        if validity == 1:
            category = determine_news_category(title, content)
            st.error(f"🚫 **Fake News** — Category: _{category}_")
        else:
            st.success("✅ **Real News**")