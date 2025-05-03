# import streamlit as st
# import pandas as pd
# from pathlib import Path

# st.title("📊 Dataset Viewer")

# # Load CSV
# data_path = Path(__file__).parent.parent / "data" / "news_testdata.csv"
# df = pd.read_csv(data_path)

# st.dataframe(df)
# # Use titles directly
# titles = df["title"].tolist()
# selected_title = st.selectbox("Select a news article to analyze:", titles)

# # Find the full article based on title
# selected_row = df[df["title"] == selected_title].iloc[0]

# # Display content preview
# st.write("**Content Preview:**")
# st.markdown(
#     f"""
#     <div style='max-height:150px; overflow-y:auto; border:1px solid #ccc; padding:10px; background:#f9f9f9; color: #333;'>
#         {selected_row['content']}
#     </div>
#     """,
#     unsafe_allow_html=True,
# )


# # Save selected row to session state
# if st.button("🔍 Analyze this"):
#     st.session_state.selected_title = selected_row["title"]
#     st.session_state.selected_content = selected_row["content"]
#     st.success("✅ Ready! Now go to the 🧪 Analyzer page from the sidebar.")

import streamlit as st
import pandas as pd
from pathlib import Path

from model import determine_news_validity, determine_news_category  # Import model functions

st.title("📊 Dataset Viewer & Analyzer")

# Load CSV
data_path = Path(__file__).parent.parent / "data" / "news_testdata.csv"
df = pd.read_csv(data_path)

# Display full dataframe
st.dataframe(df)

# Prepare dropdown options with row number and title
title_options = [f"{i}. {row['title']}" for i, row in df.iterrows()]
selected_option = st.selectbox("Select a news article to analyze:", title_options)

# Extract the selected index from the chosen dropdown value
selected_index = int(selected_option.split(".")[0])
selected_row = df.iloc[selected_index]
selected_title = selected_row["title"]
selected_content = selected_row["content"]

# Display content preview
# st.write("**Content Preview:**")
# st.markdown(
#     f"""
#     <div style='max-height:150px; overflow-y:auto; border:1px solid #ccc; padding:10px; background:#f9f9f9; color: #333;'>
#         {selected_content}
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

st.write("**Content Preview:**")
st.markdown(
    f"""
    <div style='max-height:150px; overflow-y:auto; border:1px solid #444; padding:10px; background:#2b2b2b; color:#ffffff; border-radius:5px;'>
        <em>{selected_content}</em>
    </div>
    """,
    unsafe_allow_html=True,
)



# Inline analysis
if st.button("🔍 Analyze this"):
    with st.spinner("Analyzing selected article..."):
        validity = determine_news_validity(selected_title, selected_content)
        if validity == 1:
            category = determine_news_category(selected_title, selected_content)
            st.error(f"🚫 **Fake News** — Category: _{category}_")
        else:
            st.success("✅ **Real News**")
