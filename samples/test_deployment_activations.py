"""Verify real inline HTTP activations with only the Python standard library."""

import argparse
import base64
import json
import math
import struct
from urllib.request import Request, urlopen


def fetch(url, body=None):
    request = Request(url, data=None if body is None else json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def verify(response, layers):
    params = response.get("kv_transfer_params", {})
    if params.get("hidden_states_inline") is not True:
        raise ValueError("Response did not return inline hidden states")
    tensor = params["hidden_states"]
    shape = tensor["shape"]
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError(f"Invalid hidden-state shape: {shape}")
    if shape[1] != len(layers):
        raise ValueError(f"Expected {len(layers)} layers, received {shape[1]}")
    dtype = tensor["dtype"]
    widths = {"bfloat16": 2, "float16": 2, "float32": 4}
    raw = base64.b64decode(tensor["data"], validate=True)
    if len(raw) != math.prod(shape) * widths[dtype]:
        raise ValueError("Tensor byte length disagrees with shape/dtype")
    ids = params["token_ids"]
    if ids["shape"] != [shape[0]]:
        raise ValueError("Token IDs do not match the captured token count")
    id_width = {"int64": 8, "int32": 4}[ids["dtype"]]
    if len(base64.b64decode(ids["data"], validate=True)) != shape[0] * id_width:
        raise ValueError("Token ID byte length disagrees with shape/dtype")
    nonzero = 0
    for (value,) in struct.iter_unpack({"bfloat16": "<H", "float16": "<e", "float32": "<f"}[dtype], raw):
        if dtype == "bfloat16":
            value = struct.unpack("<f", struct.pack("<I", value << 16))[0]
        if not math.isfinite(value):
            raise ValueError("Hidden states contain NaN or infinity")
        nonzero += value != 0
    if not nonzero:
        raise ValueError("All returned activations are zero")
    return {"shape": shape, "dtype": dtype, "bytes": len(raw), "nonzero_values": nonzero,
            "layers": layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model")
    parser.add_argument("--layers", default="6,13,20,27,34,43")
    parser.add_argument("--output", help="Save the full response JSON for further inspection")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    model = args.model or fetch(url + "/v1/models")["data"][0]["id"]
    response = fetch(url + "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 1, "stream": False, "n": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "kv_transfer_params": {"return_inline": True, "include_output_tokens": False},
    })
    if args.output:
        with open(args.output, "w") as output:
            json.dump(response, output)
    print(json.dumps(verify(response, [int(n) for n in args.layers.split(",")]), indent=2))
    print("PASS: HTTP response contains finite, nonzero activations")


if __name__ == "__main__":
    main()
