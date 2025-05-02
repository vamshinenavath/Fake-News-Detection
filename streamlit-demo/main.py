import streamlit as st
from model import determine_news_validity, determine_news_category

st.write("Fake News Detector")

title = st.text_input("Enter the title of the news article:")
content = st.text_area("Enter the content of the news article:")

if st.button("Check"):
    validity = determine_news_validity(title, content)

    if validity == 1:
        category = determine_news_category(title, content)
        st.write(f"The news article is Fake and belongs to the category: {category}")
    else:
        st.write("The news article is Real")