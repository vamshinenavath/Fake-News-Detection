# import os
# import sys
# import streamlit as st
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from model import determine_news_validity, determine_news_category

# st.set_page_config(page_title="🕵️ Fake News Detector", layout="wide")

# # Page title
# st.title("🧪 Fake News Analyzer")

# # Check if session state has the selected article data
# if "selected_title" in st.session_state and "selected_content" in st.session_state:
#     title = st.session_state.selected_title
#     content = st.session_state.selected_content

#     st.write(f"### Analyzing Article: {title}")
#     st.write(f"**Content:**\n{content}")

#     # Button to trigger the analysis
#     if st.button("🧪 Analyze"):
#         st.write("Analyzing the news...")

#         # Determine the validity of the news article
#         validity = determine_news_validity(title, content)
#         st.write(f"News Validity: {validity}")

#         # Determine the category of the news article
#         category = determine_news_category(title, content)
#         st.write(f"News Category: {category}")

# else:
#     st.error("No news article selected. Please go back to the Dataset Viewer and select an article.")

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
