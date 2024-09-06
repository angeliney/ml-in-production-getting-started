import numpy as np
import pandas as pd
import sys
import soundfile as sf
from streaming import MDSWriter
from torch.utils.data import DataLoader
from streaming import StreamingDataset


def load_and_process_audio_files(chunk_size=44100 * 5):
    # List to hold dictionaries with chunked audio data
    sound_arrays = []

    # Process each file
    for num in range(1, 51):
        for letter in ["a", "b"]:
            # Load the audio file
            data, sample_rate = sf.read(f"CSD/english/wav/en{num:03}{letter}.wav")
            data = np.array(data)
            # Combine stereo channels into mono
            if np.ndim(data) == 2:
                data = np.mean(data, axis=1)
            
            # Break the audio into chunks
            num_chunks = len(data) // chunk_size
            for i in range(num_chunks):
                chunk = data[i * chunk_size:(i + 1) * chunk_size]
                sound_arrays.append({
                    "audio": chunk,
                    "sample_rate": sample_rate,
                    "filename": f"en{num:03}{letter}_chunk{i+1}"
                })
    return sound_arrays


def prepare_dataset():
    """This should be run once before uploading to the cloud"""
    sound_arrays = load_and_process_audio_files()

    # Local or remote directory in which to store the compressed output files
    data_dir = 'audio-streaming-dataset'

    # A dictionary mapping input fields to their data types
    columns = {
        'audio': 'ndarray:float64:220500',
        'sample_rate': 'int',
        'filename': 'str'
    }

    # Shard compression, if any
    compression = 'zstd'

    # Save the samples as shards using MDSWriter
    with MDSWriter(out=data_dir, columns=columns, compression=compression) as out:
        for sound_array in sound_arrays:
            out.write(sound_array)

if __name__ == "__main__":
    # prepare_dataset()

    # Upload to cloud on command line:
    # gsutil -m cp -r audio-streaming-dataset gs://audio-streaming-dataset/audio-streaming-dataset
    # Before running the rest, ensure you're authenticated:
    # gcloud auth application-default login --no-launch-browser
    
    # Remote path where full dataset is persistently stored
    remote = 'gs://audio-streaming-dataset/audio-streaming-dataset'

    # Local working dir where dataset is cached during operation
    local = '/tmp/audio-streaming-dataset'

    # Create streaming dataset
    dataset = StreamingDataset(local=local, remote=remote, shuffle=True)

    # Let's see what is in sample #1337...
    sample = dataset[123]
    audio = sample['audio']
    filename = sample['filename']

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset)