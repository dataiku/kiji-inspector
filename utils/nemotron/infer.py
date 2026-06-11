import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **tokenized_chat,
    max_new_tokens=124,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False
)
print(tokenizer.decode(outputs[0]))
print("#"*32)
