import torch
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from safetensors import safe_open
import os

def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e

CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)

@torch.no_grad()
def main(
    prompt: str,
    model_folder: str = "models",
    text_encoder_device: str = "cpu",
):
    tokenizer = SD3Tokenizer()
    t5xxl = T5XXL(model_folder, text_encoder_device, torch.float32)
    clip_l = ClipL(model_folder)
    clip_g = ClipG(model_folder, text_encoder_device)

    tokens = tokenizer.tokenize_with_weights(prompt)
    l_out, l_pooled = clip_l.model.encode_token_weights(tokens["l"])
    g_out, g_pooled = clip_g.model.encode_token_weights(tokens["g"])
    t5_out, _ = t5xxl.model.encode_token_weights(tokens["t5xxl"])
    lg_out = torch.cat([l_out, g_out], dim=-1)
    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
    return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
        (l_pooled, g_pooled), dim=-1
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert text prompt(s) into SD3 conditioning tensors (c_crossattn, y), "
                    "saved one file per prompt as <out>/<prompt>.pt."
    )
    parser.add_argument(
        "prompts", nargs="*",
        help="prompt string(s) to encode; with no argument, regenerates the six paper prompts",
    )
    parser.add_argument(
        "--model-folder", default="models",
        help="folder holding clip_l/clip_g/t5xxl .safetensors (default: models/, see README)",
    )
    parser.add_argument("--out", default="conditionings", help="output folder")
    args = parser.parse_args()

    str_list = args.prompts or [
        "a fish",
        "a realistic fish",
        "a cat",
        "a releastic cat",
        "something outside",
        "someone ahead",
    ]
    os.makedirs(args.out, exist_ok=True)
    for s in str_list:
        conditioning = main(s, model_folder=args.model_folder)
        torch.save(conditioning, f"{args.out}/{s}.pt")
        print(f"Conditioning saved to {args.out}/{s}.pt")

