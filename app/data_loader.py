from datasets import load_dataset
from vision import get_image_embedding
from retrieval import build_faiss_index, find_similar
import pandas as pd
import re

def extract_age(spec):
    if not isinstance(spec, str):
        return None
    
    match = re.search(r'(\d+)-(\d+)\s*years', spec)
    if match:
        return match.group(0)
    
    return None

def load_and_clean_data():
    dataset = load_dataset("EmbeddingStudio/amazon-products-with-images", split="train")
    
    # Convert to pandas
    df = dataset.to_pandas()
    
    # Keep only Toys & Games
    df = df[df["Category"].str.contains("Toys & Games", na=False)]
    
    # Extract useful fields
    df_clean = pd.DataFrame({
        "name": df["Product Name"],
        "category": df["Category"],
        "description": df["Description"],
        "price": df["Selling Price"],
        "image": df["Image"],
        "raw_image": df["Raw Image"],
        "age": df["Product Specification"].apply(extract_age)
    })
    
    # Drop missing descriptions/images
    df_clean = df_clean.dropna(subset=["description", "image"])
    
    # Limit dataset size (IMPORTANT for speed)
    df_clean = df_clean.sample(n=300, random_state=42)
    
    print("Final dataset size:", len(df_clean))
    print("\nSample cleaned item:\n", df_clean.iloc[0])

    # 🔥 ADD THIS PART RIGHT HERE
    sample_image = df_clean.iloc[0]["raw_image"]

    embedding = get_image_embedding(sample_image)

    print("\nEmbedding shape:", embedding.shape)
    print("First 5 values:", embedding[:5])

    print("\n--- Building FAISS Index ---")
    index, embeddings = build_faiss_index(df_clean)

    print("\n--- Testing Similarity Search ---")
    query_image = df_clean.iloc[0]["raw_image"]

    results = find_similar(query_image, df_clean, index)

    print("\nTop similar toys:\n")
    print(results[["name", "category"]])

    return df_clean

if __name__ == "__main__":
    load_and_clean_data()