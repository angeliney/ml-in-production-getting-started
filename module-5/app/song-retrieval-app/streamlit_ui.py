import sys
import os
import streamlit as st
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))

from run_app import retrieve_similar_songs, setup, query_song
from audio_recorder_streamlit import audio_recorder
from librosa.util import buf_to_float


def main():
    st.title('Song retrieval app')
    sumstat_tbl, db_tbl = setup()

    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=10.0)

    if audio_bytes and len(audio_bytes) > 44:
        audio_array = np.array(buf_to_float(audio_bytes, n_bytes=4))
        st.audio(audio_array, sample_rate=44100)

        data_load_state = st.text('Querying similar songs...')
        data = retrieve_similar_songs(audio_array, sumstat_tbl, "cosine")
        # Notify the reader that the data was successfully loaded.
        data_load_state.text('Querying...done!')

        first_song = data.iloc[0]
        first_retrieved_song = query_song(db_tbl, first_song["song_num"], first_song["song_version"], first_song["chunk_num"])
        st.audio(np.array(first_retrieved_song.iloc[0]["vector"]), sample_rate=first_retrieved_song.iloc[0]["sample_rate"])
        
        second_song = data.iloc[1]
        second_retrieved_song = query_song(db_tbl, second_song["song_num"], second_song["song_version"], second_song["chunk_num"])
        st.audio(np.array(second_retrieved_song.iloc[0]["vector"]), sample_rate=second_retrieved_song.iloc[0]["sample_rate"])
        
        third_song = data.iloc[2]
        third_retrieved_song = query_song(db_tbl, third_song["song_num"], third_song["song_version"], third_song["chunk_num"])
        st.audio(np.array(third_retrieved_song.iloc[0]["vector"]), sample_rate=third_retrieved_song.iloc[0]["sample_rate"])
        
        st.subheader('Raw data')
        st.write(data)

if __name__ == "__main__":
    main()