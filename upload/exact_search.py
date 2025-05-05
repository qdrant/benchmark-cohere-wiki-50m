import json
import tqdm

from prepare_data import client
from prepare_data import read_data, DATASETS, QDRANT_COLLECTION_NAME, LIMIT, EXACT_QUERY_COUNT
from qdrant_client import models


def run_exact_search(output_file: str = "search_result_embeddings.jsonl"):
    """
    Runs exact search against the Qdrant collection.
    Saves the result in vector-db-benchmark compatible format.
    """
    points = read_data(DATASETS, limit=EXACT_QUERY_COUNT)

    print(f"Saving to {output_file}...")

    batch = []

    for point in tqdm.tqdm(points, desc="Running exact search"):
        vector = point.vector
        batch.append(vector)


    responses = client.query_batch_points(
        collection_name=QDRANT_COLLECTION_NAME,
        requests=[
            models.QueryRequest(
                query=vector,
                limit=LIMIT,
                params=models.SearchParams(exact=True),
            )
            for vector in batch
        ],
        timeout=3600
    )

    with open(output_file, "w", encoding="utf-8") as output_f:
        for vector, response in zip(batch, responses):
            hits = response.points

            record = {
                "query": vector,
                "closest_ids": [hit.id for hit in hits],
                "closest_scores": [hit.score for hit in hits],
                "conditions": None
            }

            output_f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    run_exact_search()
