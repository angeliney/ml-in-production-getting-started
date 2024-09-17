# Module 3 practice 

We've done the following experiments in `experiments/experiments.ipynb`:
- Types of retrieval: l2, cosine, dot
- Types of embedding: none, audio engineered features (MFCC, chroma, spectogram), pre-trained models (Wav2Vec2, hubert, OpenAI Whisper tiny)

Based on the performance chart below, the summary statistics of the engineered features with cosine works the best.
![performance](./experiments_performance.png)

Link to wandb project: https://wandb.ai/angeliney-georgian/children-song-dataset-retrieval?nw=nwuserangeliney 

To replicate with docker, please do the following:
1. Set up your GCP keys by running `gcloud iam service-accounts keys create gcp_keys.json --iam-account=[YOUR-IAM-ACCOUNT]`
2. `cp .env.template .env` and fill out your wandb api key.