import argparse

import torch
import numpy as np


GTEN_MAGIC_NUMBER = 0x454c49464e455447
BYTEORDER = "little"

# unsigned int to bytes.
def itob(integer, width=4):
    return int.to_bytes(integer, width, BYTEORDER, signed=True)

# float to bytes
def ftob(floatv):
    return np.array([floatv]).astype(np.float32).tobytes()



# Each row of a weight tensor is divided into blocks which are then
# quantized. Each block contains `block-size` numbers. The higher the
# block-size, the higher the compression but the model performance in
# terms of perplexity may decrease.
def quantize(t: torch.Tensor, q_blk_size: int = 64):
    assert len(t.shape) == 2, f"Illegal shape: {t.shape}"
    # 2-D tensors transposed. (d_out, d_in)
    d_in = t.shape[1]
    assert d_in % q_blk_size == 0, f"Illegal d_in: {d_in}"

    # reshape to d_out, D, Q8_BLOCK_SIZE => Q8_BLOCK_SIZE, d_out*D
    d_out = t.shape[0]
    n_blocks_per_row = d_in // q_blk_size
    n_blocks = n_blocks_per_row * d_out
    t = t.view(n_blocks, q_blk_size)

    # compute deltas for each block.
    absmax = t.abs().amax(dim=1)
    deltas = absmax / 127.0
    deltas = deltas.to(torch.float32)

    scalars = deltas.clone().view(n_blocks)
    # mask indices prevents division by zero by dividing only non-zero values.
    non_zero_idxs = scalars != 0
    scalars[non_zero_idxs] = 1.0 / scalars[non_zero_idxs]
    scalars = scalars.view(n_blocks, 1)

    # cvt
    t = torch.round(t * scalars).to(torch.int8)

    t = t.view(d_out, d_in)

    return deltas.to(torch.float16), t


def write_layer(fout, name: str, w0: torch.Tensor, dtype: str):
    name = name.encode()
    # <layer_name_size, layer_name>
    fout.write(itob(len(name)))
    fout.write(name)

    w0_name = f"{name.decode()}.weight".encode()
    fout.write(itob(len(w0_name)))
    fout.write(w0_name)

    if dtype == "fp16":
        w0 = w0.to(torch.float16)
    elif dtype == "qint8":
        assert w0.ndim == 2
        deltas, w0 = quantize(w0)
        deltas_bytes = deltas.numpy().flatten().tobytes()
        fout.write(itob(len(deltas_bytes), width=4))
        fout.write(deltas_bytes)
    else:
        assert(False)
    
    w0 = w0.numpy().flatten()
    w0_bytes = w0.tobytes()
    fout.write(itob(len(w0_bytes)))
    fout.write(w0_bytes)
    

def convert_model_to_gten(dtype):
    model_path = "tinyllama.pt"

    with open(model_path, "rb") as fin:
        ckpt = torch.load(fin)

    if dtype == "fp16":
        out_model_path = "tinyllama.fp16.gten"
    else:
        out_model_path = f"tinyllama.q8.gten"

    with open(out_model_path, "wb") as fout:
        fout.write(itob(GTEN_MAGIC_NUMBER, width=8))
        
        print("Converting wte")
        name = "model.embed_tokens.weight"
        write_layer(fout, name, w0=ckpt[name], dtype=dtype)
        
        n_layer = 22
        for i in range(n_layer):
            print(f"Converting block_{i}")

            blk_name = f"model.layers.{i}"

            name = f"{blk_name}.self_attn.q_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.self_attn.k_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.self_attn.v_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.self_attn.o_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.mlp.gate_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.mlp.up_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.mlp.down_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.input_layernorm.weight"
            write_layer(fout, name, w0=ckpt[name], dtype="fp16")

            name = f"{blk_name}.post_attention_layernorm.weight"
            write_layer(fout, name, w0=ckpt[name], dtype="fp16")
        
        print("Converting norm")
        write_layer(fout, "model.norm.weight", w0=ckpt["model.norm.weight"], dtype="fp16")

        print("Converting lm_head")
        write_layer(fout, "lm_head.weight", w0=ckpt["lm_head.weight"], dtype=dtype)


parser = argparse.ArgumentParser()
parser.add_argument("dtype", help="Model name to be converted.", choices=("fp16", "qint8"))

args = parser.parse_args()
convert_model_to_gten(args.dtype)

