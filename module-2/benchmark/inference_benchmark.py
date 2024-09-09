import soundfile as sf
import time
import multiprocessing
import numpy as np
import pandas as pd


def load_and_process_audio_files(chunk_size=44100 * 5):
    # List to hold dictionaries with chunked audio data
    sound_arrays = []

    # Process each file
    for num in range(1, 51):
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
                    "audio": chunk,
                    "sample_rate": sample_rate,
                    "filename": f"en{num:03}{letter}_chunk{i+1}"
                })
    df = pd.DataFrame(sound_arrays)
    return df


# Define the inference task to find the closest match using Euclidean distance
def find_closest_match(input_array, db_vector):
    # Calculate Euclidean distance
    distance = np.linalg.norm(input_array - db_vector)
    return distance

# Function to run inference with a single process
def run_single_process(input_array, df):
    start_time = time.time()
    # Extract vectors and filenames
    vectors = df['audio'].to_list()
    filenames = df['filename'].to_list()
    # Find the closest match
    distances = [find_closest_match(input_array, vector) for vector in vectors]
    closest_index = np.argmin(distances)
    closest_filename = filenames[closest_index]
    closest_distance = distances[closest_index]
    end_time = time.time()
    return closest_filename, closest_distance, end_time - start_time

# Function to run inference with multiple processes
def run_multiple_processes(input_array, df, num_processes):
    start_time = time.time()
    # Extract vectors and filenames
    vectors = df['audio'].to_list()
    filenames = df['filename'].to_list()
    
    with multiprocessing.Pool(num_processes) as pool:
        distances = pool.starmap(find_closest_match, [(input_array, vector) for vector in vectors])
    
    closest_index = np.argmin(distances)
    closest_filename = filenames[closest_index]
    closest_distance = distances[closest_index]
    
    end_time = time.time()
    return closest_filename, closest_distance, end_time - start_time

# Main function to benchmark performance
def benchmark_inference(input_array, df, num_processes):
    # Single process
    single_filename, single_distance, single_time = run_single_process(input_array, df)
    print(f"Single process time: {single_time:.2f} seconds")
    
    # Multiple processes
    multiple_filename, multiple_distance, multiple_time = run_multiple_processes(input_array, df, num_processes)
    print(f"Multiple processes time: {multiple_time:.2f} seconds")
    
    # Report differences
    print(f"Single process closest filename: {single_filename}, distance: {single_distance}")
    print(f"Multiple processes closest filename: {multiple_filename}, distance: {multiple_distance}")

if __name__ == '__main__':
    chunk_size = 44100 * 5
    df = load_and_process_audio_files(chunk_size=chunk_size)
    
    # Input array
    input_array = np.random.random(chunk_size)

    # Benchmark with 4 processes
    benchmark_inference(input_array, df, num_processes=4)
