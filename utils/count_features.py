import json

path = "output/activations/feature_descriptions.json"

with open(path, "r") as f:
    feature_descriptions = json.load(f)

print(f"Found {len(feature_descriptions)} feature_descriptions")
