# import streamlit as st
# import pandas as pd
# from pathlib import Path

# st.title("📊 Dataset Viewer")

# data_path = Path(__file__).parent.parent / "data" / "news_testdata.csv"

# # Load the data
# df = pd.read_csv(data_path)

# st.dataframe(df)

import streamlit as st
import pandas as pd
from pathlib import Path

st.title("📊 Dataset Viewer")

# Load CSV
data_path = Path(__file__).parent.parent / "data" / "news_testdata.csv"
df = pd.read_csv(data_path)

st.dataframe(df)
# Use titles directly
titles = df["title"].tolist()
selected_title = st.selectbox("Select a news article to analyze:", titles)

# Find the full article based on title
selected_row = df[df["title"] == selected_title].iloc[0]

# Display content preview
st.write("**Content Preview:**")
st.markdown(
    f"""
    <div style='max-height:150px; overflow-y:auto; border:1px solid #ccc; padding:10px; background:#f9f9f9; color: #333;'>
        {selected_row['content']}
    </div>
    """,
    unsafe_allow_html=True,
)


# Save selected row to session state
if st.button("🔍 Analyze this"):
    st.session_state.selected_title = selected_row["title"]
    st.session_state.selected_content = selected_row["content"]
    st.success("✅ Ready! Now go to the 🧪 Analyzer page from the sidebar.")
