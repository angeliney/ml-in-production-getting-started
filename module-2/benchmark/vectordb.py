import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
import soundfile as sf


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
                    "vector": chunk,
                    "sample_rate": sample_rate,
                    "filename": f"en{num:03}{letter}_chunk{i+1}"
                })
    return sound_arrays


if __name__ == "__main__":
    sound_arrays = load_and_process_audio_files()

    uri = "data/audio-lancedb"
    db = lancedb.connect(uri)
    
    print("Creating table...")
    tbl = db.create_table("audio_dataset", data=sound_arrays[:10])
    # Got killed here when using the full list
    
    print("Searching for similar vectors...")
    input_array = np.random.random(44100*5)
    tbl.search(input_array).limit(2).to_pandas()