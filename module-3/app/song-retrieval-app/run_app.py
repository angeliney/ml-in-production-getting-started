import os
import pandas as pd
import numpy as np 
import lancedb
import librosa

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


def setup():
    uri = "gs://children_song_retrieval/audio-lancedb"
    db = lancedb.connect(uri)
    sumstat_tbl = db.open_table("audio_feat_eng_sumstat")
    return sumstat_tbl


def retrieve_similar_songs(query_vector, db_tbl, metric="cosine"):
    embedded_vector = extract_features(query_vector, aggregate="summary_stat")
    return search(embedded_vector, sumstat_tbl, search_metric).to_pandas()


if __name__ == '__main__':
    search_metric = "cosine"
    sumstat_tbl = setup()
    query_vector = np.random.rand(44100*10)
    print(retrieve_similar_songs(query_vector, sumstat_tbl, search_metric))