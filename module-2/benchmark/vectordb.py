import time
import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
import soundfile as sf


def load_and_process_audio_files(file_range=(1, 11), chunk_size=44100 * 5):
    # List to hold dictionaries with chunked audio data
    sound_arrays = []

    # Process each file
    for num in range(file_range[0], file_range[1]):
        for letter in ["a", "b"]:
            # Load the audio file
            data, sample_rate = sf.read(f"./data/CSD/english/wav/en{num:03}{letter}.wav")
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
    file_chunks = 10
    chunk_size = 44100 * 10
    
    uri = "data/lancedb-data/audio-lancedb"
    db = lancedb.connect(uri)
    
    for i in range(1, 51, file_chunks):
        print(f"Processing files {i} to {i + file_chunks}...")
        sound_arrays = load_and_process_audio_files(file_range=(i, i + file_chunks), chunk_size=chunk_size)
        if i == 1:
            tbl = db.create_table("audio_dataset", data=sound_arrays)
        else:
            tbl.add(sound_arrays)
    
    print("Searching for similar vectors...")
    input_array = np.random.random(chunk_size)
    start_time = time.time()
    tbl.search(input_array).limit(2).to_pandas()
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")