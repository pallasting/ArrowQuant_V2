#!/usr/bin/env python3
"""Check what weights are in the converted model."""

import pyarrow.parquet as pq

# Read the parquet file
table = pq.read_table("D:/ai-models/clip-vit-b32/weights.parquet")

# Get layer names
layer_names = table.column("layer_name").to_pylist()

print(f"Total layers: {len(layer_names)}")
print("\nSearching for projection layers:")

for name in layer_names:
    if "projection" in name.lower():
        print(f"  - {name}")

print("\nFirst 10 layers:")
for i, name in enumerate(layer_names[:10]):
    print(f"  {i+1}. {name}")

print("\nLast 10 layers:")
for i, name in enumerate(layer_names[-10:]):
    print(f"  {len(layer_names)-9+i}. {name}")
