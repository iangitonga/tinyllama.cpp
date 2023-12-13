#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "tinyllama": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.fp16.gten",
        "q8": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.q8.gten",
    }
}

DTYPES = (
    "fp16"
)


def _download_model(url, model_path):
    print("Downloading ...")
    with request.urlopen(url) as source, open(model_path, "wb") as output:
        download_size = int(source.info().get("Content-Length"))
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
            progress_perc = int(len(buffer) / download_size * 100.0)
            # trick to make it work with jupyter.
            print(f"\rDownload Progress: {progress_perc}%", end="", flush=True)
    print("")

def download_model(dtype):
    model_name = "tinyllama"
    if dtype == "q8":
        model_path = os.path.join("models", f"{model_name}.q8.gten")
    else:
        model_path = os.path.join("models", f"{model_name}.fp16.gten")

    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    model_url_key = f"{model_name}.q8" if dtype == "q8" else f"{model_name}.fp16"
    _download_model(MODELS_URLS[model_name][dtype], model_path)

# python model.py model inf
if len(sys.argv) != 2:
    print(f"Args provided: {sys.argv}")
    print("DTYPE is one of (fp16, q8)")
    exit(-1)


try:
    download_model(sys.argv[1])
except:
    exit(-2)
