import streamlit as st
from data_loader import load_and_clean_data
from retrieval import build_faiss_index, find_similar
from PIL import Image

# 🔥 Make layout wider (UI polish)
st.set_page_config(layout="wide")

st.title("🧸 Toy Intelligence Assistant")
st.markdown("Upload a toy image to discover similar products and insights.")

@st.cache_resource
def load_system():
    df = load_and_clean_data()
    index, embeddings = build_faiss_index(df)
    return df, index

df, index = load_system()

uploaded_file = st.file_uploader("Upload a toy image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    results = find_similar(image, df, index)

    st.subheader("🔍 Similar Toys")

    # 🔥 Slightly improved layout (2-column grid, still simple)
    cols = st.columns(2)

    for idx, (i, row) in enumerate(results.iterrows()):
        col = cols[idx % 2]

        with col:
            st.image(image, caption="Uploaded Image")

            st.markdown(f"**{row['name']}**")
            st.caption(row["category"])

            insights = f"""
            **Description:** {row['description'][:150]}...

            **Recommended Age:** {row.get('age', 'Unknown')}
            """

            from explainer import explain_match

            with st.spinner("Explaining why this matches..."):
                explanation = explain_match(image, row)

            st.markdown(insights)
            st.markdown("### 🧠 Why this matches")
            st.write(explanation)
            st.markdown("---")

    # ✅ Chatbot stays OUTSIDE loop (unchanged logic)
    st.markdown("## 💬 Ask about these toys")

    user_question = st.text_input(
        "Ask a question (e.g., Is this good for a 6-year-old?)",
        key="toy_question_input"
    )

    if user_question:
        from rag import ask_toy_question

        answer = ask_toy_question(user_question, results)

        st.markdown("### 🤖 Answer")
        st.write(answer)