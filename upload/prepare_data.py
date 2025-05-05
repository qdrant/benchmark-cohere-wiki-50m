import json
import os
from typing import Iterable

import tqdm

from hf import read_dataset_stream
from qdrant_client import QdrantClient, models

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = "benchmark"
EXACT_QUERY_COUNT = 1000
LIMIT=100
LIMIT_POINTS = 50_000_000
DATASETS = [
    "Cohere/wikipedia-22-12-simple-embeddings",
    "Cohere/wikipedia-22-12-en-embeddings",
    "Cohere/wikipedia-22-12-de-embeddings"
]


client = QdrantClient(
    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    timeout=3600 # For full-scan search
)


def create_collection(force_recreate=False):
    if force_recreate:
        client.delete_collection(QDRANT_COLLECTION_NAME)

    if client.collection_exists(QDRANT_COLLECTION_NAME):
        return

    client.create_collection(
        QDRANT_COLLECTION_NAME,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            )
        ),
        hnsw_config=models.HnswConfigDiff(
            m=32,
            ef_construct=256,
        ),
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
            on_disk=True,
            datatype=models.Datatype.FLOAT16,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            max_segment_size=50_000_000
        )
    )


def read_data(
        datasets: list[str],
        skip_first: int = 0,
        limit: int = LIMIT_POINTS
) -> Iterable[models.PointStruct]:
    
    n = 0
    for dataset in datasets:
        stream = read_dataset_stream(dataset, split="train")
        for item in stream:
            n += 1

            if n <= skip_first:
                continue

            if n >= limit:   
                return

            embedding = item.pop("emb")

            yield models.PointStruct(
                id=n,
                vector=embedding.tolist(),
                payload=item,
            )


def load_all():

    # Use first 1000 points for testing
    skip_first = EXACT_QUERY_COUNT

    points = read_data(DATASETS, skip_first=skip_first, limit=LIMIT_POINTS + skip_first)

    client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME,
        points=tqdm.tqdm(points, desc="Uploading points"),
        parallel=8,
        batch_size=64,
    )

def main():
    create_collection(force_recreate=True)
    load_all()


if __name__ == "__main__":
    main()
