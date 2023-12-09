#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "tinyllama": {
        "float16": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.fp16.gten",
    }
}

DTYPES = (
    "float16"
)


def show_progress(cur_size, max_size):
    ls = [" "] * 50
    prog = int(cur_size / max_size * 50)
    for i in range(prog):
        ls[i] = "#"
    print("Progress: [" + "".join(ls) + "]", end="\r", flush=True)
    if cur_size == max_size:
        print()

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
            print(f"DOwnload Progress(%):  {progress_perc}")

def download_model(model_name, dtype):
    # if dtype == "qint8":
    #     model_path = os.path.join("models", f"{model_name}.q8.gten")
    # else:
    model_path = os.path.join("models", f"{model_name}.fp16.gten")
    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    model_url_key = f"{model_name}.qint8" if dtype == "qint8" else f"{model_name}.fp16"
    _download_model(MODELS_URLS[model_name][dtype], model_path)

# python model.py model inf
# if len(sys.argv) != 1:
#     print(f"Args provided: {sys.argv}")
#     print("usage: model_registry.py")
#     # print("DTYPE is one of (float16, qint8)")
#     exit(-1)


try:
    download_model("tinyllama", "float16")
except:
    exit(-2)
