import os
import pandas as pd
import numpy as np 
import lancedb
import wandb
import librosa
import torch
from dotenv import load_dotenv
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, AutoProcessor, AutoModel, WhisperFeatureExtractor
)


def test_method(test_fn, embed_fn=None):
    queries_tbl = db.open_table("audio_example_queries")
    total_rows = queries_tbl.count_rows()
    song_num_actual = []
    song_num_retrieved = []

    conditions = [
        "(offset == 0) and (pitch_shift == 0) and (time_stretch == 1.0)",
        "(offset != 0) and (pitch_shift == 0) and (time_stretch == 1.0)",
        "(offset == 0) and (pitch_shift != 0) and (time_stretch == 1.0)",
        "(offset == 0) and (pitch_shift == 0) and (time_stretch != 1.0)",
        "(offset == 0) and (pitch_shift != 0) and (time_stretch != 1.0)",
        "(offset != 0) and (pitch_shift != 0) and (time_stretch != 1.0)",
    ]

    for condition in conditions:
        print(f"Running test for condition: {condition}")
        filtered_tbl = queries_tbl.search().where(condition).select(["song_num", "vector"])

        for _, row in filtered_tbl.to_pandas().iterrows():
            if embed_fn:
                row["vector"] = embed_fn(row["vector"])
            song_num_actual.append(row["song_num"])
            retrieved_info_list = test_fn(row["vector"]).to_pandas()

            song_num_retrieved.append([retrieved_info["song_num"] 
                                       for _, retrieved_info in retrieved_info_list.iterrows()])
    return song_num_actual, song_num_retrieved


def calculate_mrr(actual_songs, retrieved_songs):
    """
    Calculate Mean Reciprocal Rank (MRR) for a list of song retrievals.

    Parameters:
    actual_songs (list of int): A list of the actual song numbers.
    retrieved_songs (list of list of int): A list of lists, where each inner list contains retrieved song numbers.

    Returns:
    float: The Mean Reciprocal Rank (MRR) score.
    """
    reciprocal_ranks = []

    for actual, retrieved in zip(actual_songs, retrieved_songs):
        try:
            # Find the rank (1-indexed) of the actual song in the retrieved list
            rank = retrieved.index(actual) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            # If the actual song is not in the retrieved list, reciprocal rank is 0
            reciprocal_ranks.append(0.0)

    # Calculate the mean of the reciprocal ranks
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def retrieval_recall(actual_songs, retrieved_songs):
    in_retrieved = []
    for actual, retrieved in zip(actual_songs, retrieved_songs):
        in_retrieved.append(actual in retrieved)
    
    return np.sum(in_retrieved)/len(in_retrieved)


def test_and_log(search_fn, embed_fn, search_metric, embedding):
    actual, retrieved = test_method(search_fn, embed_fn)
    mrr = calculate_mrr(actual, retrieved)
    rr = retrieval_recall(actual, retrieved) 
    print("mrr", mrr, "rr", rr)

    wandb.init(
        # set the wandb project where this run will be logged
        project="children-song-dataset-retrieval",

        # track hyperparameters and run metadata
        config={
        "embedding": embedding,
        "retrieval": search_metric,
        }
    )

    wandb.log({"mrr": mrr})
    wandb.log({"retrieval_recall": rr})
    wandb.finish()


def search(query_vector, db_tbl, metric="l2"):
    return db_tbl.search(query_vector).metric(metric).limit(3)


def extract_features(audio, sr=44100, aggregate="summary_stat"):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Extract Mel-scaled spectrogram features
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

    if aggregate == "summary_stat":
        # Aggregate the MFCCs across time
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfcc_embedding = np.concatenate([mfccs_mean, mfccs_std])

        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_embedding = np.concatenate([chroma_mean, chroma_std])

        mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)
        mel_spectrogram_std = np.std(mel_spectrogram, axis=1)
        mel_spectrogram_embedding = np.concatenate([mel_spectrogram_mean, mel_spectrogram_std])
    
    else:
        # Flatten the MFCCs into a 1D array
        mfcc_embedding = mfccs.flatten()
        chroma_embedding = chroma.flatten()
        mel_spectrogram_embedding = mel_spectrogram.flatten()
   
    return np.concatenate([mfcc_embedding, chroma_embedding, mel_spectrogram_embedding])


def embed_lookup_data(embed_fn, df, db_name):
    ## Re-embed the data with the new features
    db_setup = False

    batch_size = len(df)//5
    for i in range(0, len(df), batch_size):
        print(i)
        sound_arrays = []
        for _, row in df.iloc[i:i+batch_size].iterrows():
            sound_arrays.append(
                {
                    "vector": embed_fn(row["vector"]),
                    "sample_rate": row["sample_rate"],
                    "offset": row["offset"],
                    "pitch_shift": row["pitch_shift"],
                    "time_stretch": row["time_stretch"],
                    "song_num": row["song_num"],
                    "song_version": row["song_version"],
                    "chunk_num": row["chunk_num"],
                    "filename": row["filename"],
                }
            )
    

        if db_setup:
            feat_tbl.add(sound_arrays)
        else:
            feat_tbl = db.create_table(db_name, data=sound_arrays)
            db_setup = True



def embed_with_model(audio, processor, model, output_field=None):
    y_16k = librosa.resample(audio, orig_sr=44100, target_sr=16000)

    # Preprocess the audio for the model
    inputs = processor(y_16k, sampling_rate=16000, return_tensors="pt", padding=True)

    # Pass the inputs to the model
    with torch.no_grad():
        out = model(**inputs)

    if output_field:
        out = out[output_field] #last_hidden_state or extract_features

    return out


if __name__ == '__main__':
    load_dotenv(".env")
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # uri = "../../data/lancedb-data/audio-lancedb"
    uri = "gs://children_song_retrieval/audio-lancedb"
    db = lancedb.connect(uri)
    db_tbl = db.open_table("audio_dataset")
    audio_df = db_tbl.to_pandas()

    search_metrics = ["l2", "cosine", "dot"]

    # Experiment without any embedding
    print("Experiment without any embedding")
    for search_metric in search_metrics:
        print(search_metric)
        test_and_log(lambda x: search(x, db_tbl, search_metric),
                    None,
                    search_metric,
                    "none")
        
    # Experiment with engineered features: MFCC, Chroma, Mel-spectrogram - flattened or summary statistics
    # First, extract the features and embed them
    print("Experiment with engineered features")
    if "audio_feat_eng_sumstat" not in db.table_names():
        embed_lookup_data(
            lambda audio: extract_features(audio, aggregate="summary_stat"),
            audio_df,
            "audio_feat_eng_sumstat"
        )

    if "audio_feat_eng_full" not in db.table_names():
        embed_lookup_data(
            lambda audio: extract_features(audio, aggregate="full"),
            audio_df,
            "audio_feat_eng_full"
        )

    sumstat_tbl = db.open_table("audio_feat_eng_sumstat")
    fullfeat_tbl = db.open_table("audio_feat_eng_full")

    # Test the search with the new embeddings
    for search_metric in search_metrics:
        print(search_metric)
        test_and_log(
            lambda x: search(x, sumstat_tbl, search_metric),
            lambda audio: extract_features(audio, aggregate="summary_stat"),
            search_metric,
            "audio_feat_eng_sumstat"
        )

        test_and_log(
            lambda x: search(x, fullfeat_tbl, search_metric),
            lambda audio: extract_features(audio, aggregate="full"),
            search_metric,
            "audio_feat_eng_full"
        )

    # Try with pretrained models
    # We're going to extract the last_hidden_state, although technically can also experiment with extract_features
    models = ["facebook/wav2vec2-base-960h", "facebook/hubert-large-ls960-ft", "openai/whisper-tiny"]
    
    print("Experiment with pretrained models")
    for model_name in models:
        print(model_name)
        if "openai" not in model_name:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2Model.from_pretrained(model_name)

        else:
            whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            whisper_processor = AutoProcessor.from_pretrained(model_name)
            def processor(audio, sampling_rate=16000, return_tensors="pt", **kwargs):
                return whisper_processor.feature_extractor.pad([{
                        "input_features":
                        whisper_feature_extractor(audio, sampling_rate=sampling_rate)["input_features"][0]}],
                    return_tensors=return_tensors)

            whisper_model = AutoModel.from_pretrained(model_name)
            def model(input_features):
                return whisper_model.encoder(input_features)
            
        db_name = f"audio_feat_{model_name.replace('/', '_').replace("-", "_")}_last_hidden_state"
        if db_name not in db.table_names():
            embed_lookup_data(
                lambda audio: embed_with_model(audio, processor, model, output_field="last_hidden_state").numpy().flatten(),
                audio_df,
                db_name
            )

        hidden_state_tbl = db.open_table(db_name)

        for search_metric in search_metrics:
            print(search_metric)
            test_and_log(
                lambda x: search(x, hidden_state_tbl, search_metric),
                lambda audio: embed_with_model(audio, processor, model, output_field="last_hidden_state").numpy().flatten(),
                search_metric,
                db_name
            ) 


