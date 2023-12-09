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


def tensor_to_bytes(t, out_dtype):
    return t.detach().numpy().flatten().astype(out_dtype).tobytes()


def write_layer(fout, name, w0):
    name = name.encode()
    # <layer_name_size, layer_name>
    fout.write(itob(len(name)))
    fout.write(name)

    w0_name = f"{name.decode()}.weight".encode()
    fout.write(itob(len(w0_name)))
    fout.write(w0_name)

    w0 = w0.to(torch.float16)
    
    w0 = w0.numpy().flatten()
    w0_bytes = w0.tobytes()
    fout.write(itob(len(w0_bytes)))
    fout.write(w0_bytes)
    

def convert_model_to_gten():
    model_path = "tinyllama.pt"

    with open(model_path, "rb") as fin:
        ckpt = torch.load(fin)

    with open("tinyllama.fp16.gten", "wb") as fout:
        fout.write(itob(GTEN_MAGIC_NUMBER, width=8))
        
        print("Converting wte")
        write_layer(fout, "model.embed_tokens.weight", w0=ckpt["model.embed_tokens.weight"])
        
        n_layer = 22
        for i in range(n_layer):
            print(f"Converting block_{i}")

            blk_name = f"model.layers.{i}"

            name = f"{blk_name}.self_attn.q_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.self_attn.k_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.self_attn.v_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.self_attn.o_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.mlp.gate_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.mlp.up_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.mlp.down_proj.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.input_layernorm.weight"
            write_layer(fout, name, w0=ckpt[name])

            name = f"{blk_name}.post_attention_layernorm.weight"
            write_layer(fout, name, w0=ckpt[name])
        
        print("Converting norm")
        write_layer(fout, "model.norm.weight", w0=ckpt["model.norm.weight"])

        print("Converting lm_head")
        write_layer(fout, "lm_head.weight", w0=ckpt["lm_head.weight"])


convert_model_to_gten()
