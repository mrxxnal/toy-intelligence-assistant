import faiss
import numpy as np
from vision import get_image_embedding


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

    # Create FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("FAISS index built successfully")

    return index, embeddings


def find_similar(query_image, df, index, top_k=5):
    query_embedding = get_image_embedding(query_image).astype("float32").reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]]

    return results