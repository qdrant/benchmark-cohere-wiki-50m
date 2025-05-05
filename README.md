
# Bechmarkign Qdrant on Cohere Dataset

Cohere datasets used:

* [wikipedia-22-12-en-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings) - 35.2M rows
* [wikipedia-22-12-simple-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings) - 486k rows
* [wikipedia-22-12-de-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-de-embeddings) - 15M rows


Total of 50M embeddings of 768 dimensions.
Target accuracy: 0.99


## Scripts

* `upload/prepare_data.py` - downloads data from Hugging Face, file per file, and uploads it to Qdrant.
* `upload/hf.py` - helper functions to work with Hugging Face datasets.
* `upload/exact_search.py` - use full-scan queries to generate ground truth for the benchmark (should be used with enough RAM and **full-precision** embeddings).

## Hardware

 - AWS `r6id.4xlarge` - 16 vCPUs, 128 GB RAM, Disk: 950 GB Local NVMe SSD. 


## Collection configuration

Here is a breakdown of the collection configuration:

```python
client.create_collection(
    QDRANT_COLLECTION_NAME,

    # In our experiments 768d vectors perform better with scalar quantization.
    # Binary quantization is possible, but would require more re-scoring and will likely saturate disk faster.
    #
    # Expected memory consumption: 50M rows * 768 bytes ~= 38 GB
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
            quantile=0.99,
        )
    ),
    # For higher accuracy those parameters are tuned to
    # provide better recall at cost of some longer optimization time.
    #
    # Expected memory consumption: 50M rows * 32 u32 * 2 bytes ~= 12Gb
    hnsw_config=models.HnswConfigDiff(
        m=32,
        ef_construct=256,
    ),
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        # We don't need to force original vectors to RAM
        on_disk=True,
        # Float16 is well enough precision for storing original vectors
        # There will be virtually no difference in accuracy with Float32
        datatype=models.Datatype.FLOAT16,
    ),

    # We want higher RPS, so we prefer larger segments.
    # This configuration will create default amount of segments (8),
    # each with up to 50 millions Kb of vectors.
    #
    # Bigger segments will require more indexing time.
    optimizers_config=models.OptimizersConfigDiff(
        max_segment_size=50_000_000
    )
)
```


## Generate reference data

In order to obtain ground truth for the benchmark, we need to run exact search on the dataset.
This might take a while or require a big machine with enough RAM. So we pre-generate reference data
and share it as a part of testing repository.


If you want to generate your own, you can do so with the following command:

```bash
python upload/exact_search.py
```


## Uploading data


Run qdrant on server:

```bash
# Make sure local directory is mounted to local SSD
docker run --rm -it --network=host -v $(pwd)/qdrant-storage:/qdrant/storage qdrant/qdrant:v1.14.0
```

```bash
export QDRANT_CLUSTER_URL=http://localhost:6333
export QDRANT_API_KEY=

# Run upload script
python upload/prepare_data.py
```

Upload will take about 2.5 hours, this includes downloading data from Hugging Face,
converting it to Qdrant format and uploading to Qdrant.

Total upload and indexing time is expected to be around 5 hours.


## Running search

At this point we would like to re-use existing infrastructure for running search - [vector-db-benchmark](https://github.com/qdrant/vector-db-benchmark).

Scripts from this project can work with a compatible format as `upload/exact_search.py` produces, and it is already contains all necessary parameters.

### Setup

```bash
git clone https://github.com/qdrant/vector-db-benchmark.git
cd vector-db-benchmark

# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install poetry && poetry install
```

### Clearing the cache

To avoid system disk cache contamination with benchmark-specific data, we clear disk cache for the whole system, and only after than launch Qdrant.

```bash
sudo bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches'

docker run --rm -it --network=host -v $(pwd)/qdrant-storage:/qdrant/storage qdrant/qdrant:v1.14.0
```

### Hearing-up Qdrant

To simulate production environment, we will pre-run random queries through the collection:

```bash
docker run --rm -it --network=host qdrant/bfb:dev ./bfb -n 300000 -d 768 --skip-create --skip-upload --skip-wait-index --quantization-rescore=true --search --search-hnsw-ef=256 --search-limit 100 -p 100 -t 10
```

Random queries guarantee that there are no correlation with real benchmark queries.


### Running the benchmark


```
python3 -m run --engines qdrant-rescore-only --datasets cohere-wiki-50m-test-only --skip-upload
```

It will download the reference queries, run the benchmark and save report to `results/` directory.


## Results

This is about the expected result for the benchmark:

```
{ 
  "params": {
    "dataset": "cohere-wiki-50m-test-only",
    "experiment": "qdrant-rescore-only",
    "engine": "qdrant",
    "parallel": 16,
    "config": {
      "hnsw_ef": 300,
      "quantization": {
        "rescore": true
      }
    }
  },
  "results": {
    "total_time": 2.2990338590025203,
    "mean_time": 0.035964066186201155,
    "mean_precisions": 0.9908208208208209,
    "std_time": 0.004878619847172566,
    "min_time": 0.023429110005963594,
    "max_time": 0.06853805600258056,
    "rps": 434.53035547437963,
    "p95_time": 0.04366774741065455,
    "p99_time": 0.050363237840938366
  }
}
```




