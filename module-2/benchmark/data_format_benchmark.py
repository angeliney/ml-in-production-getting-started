import time, os
import soundfile as sf
import pandas as pd
import numpy as np


def load_and_process_audio_files():
    sound_arrays = []

    for num in range(1, 5):
        for letter in ["a", "b"]:
            # print(f"Loading en{num:03}{letter}")
            data, sample_rate = sf.read(f"CSD/english/wav/en{num:03}{letter}.wav")
            sound_arrays.append({
                "audio": data,
                "sample_rate": sample_rate,
                "filename": f"en{num:03}{letter}"
            })
    
    df = pd.DataFrame(sound_arrays)

    # Split the 2d array into 2 columns of 1d array to make it easier to save to different file formats
    def split_2d_array(arr):
        np_arr = np.array(arr)
        return pd.Series([np_arr[:, 0], np_arr[:, 1]])

    # Apply the function to the 'array_column'
    df_split = df['audio'].apply(split_2d_array)

    # Rename the columns and join with the original DataFrame (excluding 'array_column')
    df_split.columns = ['audio_dim0', 'audio_dim1']
    df_final = pd.concat([df.drop('audio', axis=1), df_split], axis=1)

    df = df_final
    return df


def benchmark_save_load(df, format_name, save_func, load_func, file_ext, results):
    file_name = f'benchmark_test.{file_ext}'

    # Measure save time
    start_time = time.time()
    save_func(df, file_name)
    save_time = time.time() - start_time

    # Measure load time
    start_time = time.time()
    loaded_df = load_func(file_name)
    load_time = time.time() - start_time

    # Clean up the file after testing
    os.remove(file_name)

    # Store results
    results[format_name] = {'save_time': save_time, 'load_time': load_time}

if __name__ == '__main__': 
    df = load_and_process_audio_files()

    # Dictionary to hold benchmark results
    results = {} 

    # CSV
    benchmark_save_load(
        df,
        'CSV',
        lambda df, filename: df.to_csv(filename, index=False),
        lambda filename: pd.read_csv(filename),
        'csv',
        results
    )

    # HDF5
    benchmark_save_load(
        df,
        'HDF5',
        lambda df, filename: df.to_hdf(filename, key='df', mode='w'),
        lambda filename: pd.read_hdf(filename, 'df'),
        'h5',
        results
    )

    # Feather
    benchmark_save_load(
        df,
        'Feather',
        lambda df, filename: df.to_feather(filename),
        lambda filename: pd.read_feather(filename),
        'feather',
        results
    )

    # Parquet
    benchmark_save_load(
        df,
        'Parquet',
        lambda df, filename: df.to_parquet(filename, engine='pyarrow'),
        lambda filename: pd.read_parquet(filename),
        'parquet',
        results
    )

    # Display the results
    for fmt, times in results.items():
        print(f"{fmt}: Save time = {times['save_time']:.4f} s, Load time = {times['load_time']:.4f} s")


