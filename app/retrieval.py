import faiss
import numpy as np
from vision import get_image_embedding


# ----------------------------
# BUILD FAISS INDEX
# ----------------------------
def build_faiss_index(df):
    embeddings = []

    print("Generating embeddings for dataset...")

    for i, row in df.iterrows():
        emb = get_image_embedding(row["raw_image"])
        embeddings.append(emb)

        if i % 50 == 0:
            print(f"Processed {i} items")

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    # Normalize embeddings for better cosine-like behavior
    faiss.normalize_L2(embeddings)

    # FAISS index (cosine similarity via L2 on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("FAISS index built successfully")

    return index, embeddings


# ----------------------------
# SMART RERANKING SYSTEM
# ----------------------------
def rerank_results(df, indices, top_k=5):
    results = df.iloc[indices].copy()

    def category_boost(cat):
        if "Toys & Games" in str(cat):
            return 1.3
        return 1.0

    def age_boost(age):
        if age is None:
            return 1.0
        if "4-12" in str(age):
            return 1.2
        return 1.05

    scores = []

    for _, row in results.iterrows():
        score = 1.0

        # category relevance
        score *= category_boost(row["category"])

        # age appropriateness
        score *= age_boost(row.get("age", None))

        scores.append(score)

    results["final_score"] = scores

    results = results.sort_values("final_score", ascending=False)

    return results.head(top_k)


# ----------------------------
# SEARCH FUNCTION
# ----------------------------
def find_similar(query_image, df, index, top_k=5):
    query_embedding = get_image_embedding(query_image).astype("float32").reshape(1, -1)

    # normalize query too
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k * 3)

    # rerank instead of raw FAISS output
    results = rerank_results(df, indices[0], top_k)

    return results