# Module 2 Benchmarking
For this module, we will be doing data benchmarking exercise with the open-source children audio dataset from https://dagshub.com/kinkusuma/children-song-dataset

## Benchmark various Pandas formats in terms of data saving/loading, focusing on load time and save time.
Script: 
- data_format_benchmark.ipynb
- data_format_benchmark.py

Results:
```
CSV: Save time = 0.0029 s, Load time = 0.0011 s
HDF5: Save time = 0.8560 s, Load time = 1.7598 s
Feather: Save time = 0.9728 s, Load time = 0.1597 s
Parquet: Save time = 1.6073 s, Load time = 0.2989 s
```

## Benchmark inference performance using single and multiple processes, and report the differences in time
Script:
- inference_benchmark.ipynb
- inference_benchmark.py

Results:
```
Single process time: 5.57 seconds
Multiple processes time: 14.22 seconds
Single process closest filename: en024b_chunk11, distance: 271.2306041609267
Multiple processes closest filename: en024b_chunk11, distance: 271.2306041609267
```

## Develop code for converting your dataset into the StreamingDataset format.
Script: streaming_dataset.py


## Transform your dataset into a vector format, and utilize VectorDB for ingestion and querying
Script: vectordb.py

Results:
```
Search time: 0.6097662448883057 seconds
```
From these results, it seems to make sense to store the data in a csv format and continue with lancedb the application itself