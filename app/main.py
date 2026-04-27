import streamlit as st
from data_loader import load_and_clean_data
from retrieval import build_faiss_index, find_similar
from PIL import Image

# UI config
st.set_page_config(layout="wide")

st.title("🧸 Toy Intelligence Assistant")

mode = st.selectbox(
    "Choose Mode",
    ["🔍 Visual Finder", "🎁 Gift Advisor", "🧠 Parent Assistant"]
)

# -----------------------------
# GIFT ADVISOR INPUT
# -----------------------------
gift_query = None

if mode == "🎁 Gift Advisor":
    gift_query = st.text_input(
        "Describe the gift (age, budget, interest)"
    )

st.markdown("Upload a toy image to discover similar products and insights.")

# cache heavy pipeline
@st.cache_resource
def load_system():
    df = load_and_clean_data()
    index, embeddings = build_faiss_index(df)
    return df, index

df, index = load_system()

uploaded_file = st.file_uploader("Upload a toy image", type=["jpg", "png", "jpeg"])

# -----------------------------
# VISUAL FINDER MODE
# -----------------------------
if mode == "🔍 Visual Finder" and uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    results = find_similar(image, df, index)

    if results is None or len(results) == 0:
        st.warning("No similar toys found.")
        st.stop()

    st.subheader("🔍 Similar Toys")

    cols = st.columns(2)

    from explainer import explain_match

    for idx, (_, row) in enumerate(results.iterrows()):
        col = cols[idx % 2]

        with col:
            st.image(row["image"], caption=row["name"], width=250)

            st.markdown(f"**{row['name']}**")
            st.caption(row["category"])

            st.write(
                f"**Description:** {str(row['description'])[:150]}..."
            )

            st.write(
                f"**Recommended Age:** {row.get('age', 'Unknown')}"
            )

            with st.spinner("Explaining why this matches..."):
                explanation = explain_match(image, row)

            st.markdown("### 🧠 Why this matches")
            st.write(explanation)

            st.markdown("---")

    # -----------------------------
    # CHATBOT (OUTSIDE LOOP)
    # -----------------------------
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

# -----------------------------
# 🎁 GIFT ADVISOR MODE (ADDED)
# -----------------------------
elif mode == "🎁 Gift Advisor" and gift_query:

    st.subheader("🎁 Smart Gift Recommendations")

    # reuse FAISS index but now do lightweight text filtering
    def simple_gift_rank(df, query, top_k=5):
        query = query.lower()
        scored = []

        for _, row in df.iterrows():
            text = (str(row["name"]) + " " + str(row["description"])).lower()

            score = sum(1 for w in query.split() if w in text)

            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [r for _, r in scored[:top_k]]

    results = simple_gift_rank(df, gift_query)

    cols = st.columns(2)

    for idx, row in enumerate(results):
        col = cols[idx % 2]

        with col:
            st.image(row["image"], width=250)
            st.markdown(f"**{row['name']}**")
            st.caption(row["category"])

            st.write(str(row["description"])[:200] + "...")
            st.markdown("---")