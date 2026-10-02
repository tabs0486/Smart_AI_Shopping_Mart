import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = "vector_store/products.index"
META_PATH = "vector_store/products_metadata.pkl"


class ProductVectorStore:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def create_text(self, row):
        return (
            f"Product: {row['name']}. "
            f"Category: {row['category']}. "
            f"Source: {row['source']}. "
            f"Price: {row['price']}. "
            f"Condition: {row['condition']}. "
            f"Delivery city: {row['delivery_city']}. "
            f"Delivery days: {row['delivery_days']}. "
            f"Payment: {row['payment_method']}. "
            f"Ingredients: {row['ingredients']}. "
            f"Specifications: {row['specifications']}."
        )

    def build(self, csv_path="data/products.csv"):

        df = pd.read_csv(csv_path)

        texts = [
            self.create_text(row)
            for _, row in df.iterrows()
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        os.makedirs("vector_store", exist_ok=True)

        faiss.write_index(
            index,
            INDEX_PATH
        )

        with open(META_PATH, "wb") as f:
            pickle.dump(df, f)

    def search(self, query, k=10):

        if not os.path.exists(INDEX_PATH):
            self.build()

        index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            df = pickle.load(f)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = index.search(
            query_embedding,
            min(k, len(df))
        )

        results = df.iloc[indices[0]].copy()
        results["similarity"] = scores[0]

        return results
