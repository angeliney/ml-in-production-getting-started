import shlex
import subprocess
from pathlib import Path

import modal
# image = modal.Image.debian_slim(python_version="3.11").pip_install(
#     "streamlit~=1.35.0", "numpy~=1.26.4", "pandas~=2.2.2"
# )
image = modal.Image.from_registry("angeliney/song_retrieval_streamlit:latest")


app = modal.App(name="modal-streamlit-song-retrieval", image=image)
streamlit_script_local_path = Path(__file__).parent / "song-retrieval-app" / "streamlit_ui.py"
streamlit_script_remote_path = Path("/root/song-retrieval-app/streamlit_ui.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "Script not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)
lib_script_mount = modal.Mount.from_local_file(
    Path(__file__).parent / "song-retrieval-app" / "run_app.py",
    Path("/root/song-retrieval-app/run_app.py"),
)
@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount,
            lib_script_mount],
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)

