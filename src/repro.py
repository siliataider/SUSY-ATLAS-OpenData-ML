import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
import os

# 1. SETUP: Create a dummy Parquet file on disk
input_file = "/home/siliataider/Documents/root/data_loader_bench/SUSY-ATLAS-OpenData-ML/data/SM_SUSY_C1C1_mu_mll-15_MET_both_jets.parquet"


# 2. DATA LOADING: Hugging Face Streaming
print("####### HuggingFace DataLoader (Fast Parquet Streaming) ###############")

batch_size = 10
validation_split = 0.3

# Load in streaming mode (doesn't load file into memory)
full_dataset = load_dataset(
    "parquet",
    data_files=input_file,
    split="train",
    streaming=True
).with_format("torch")

num_samples = 1749075
# Manual split for streaming
# Note: In a real script, you'd use your known 'dataset_size' variable
num_validation_samples = int(num_samples * validation_split)
val_dataset = full_dataset.take(num_validation_samples)
train_dataset = full_dataset.skip(num_validation_samples)

# 3. THE FAST COLLATOR
# This avoids Python loops by operating on the batch dictionary directly
def fast_collate(batch):
    # batch is a list of dicts: [{'col1': tensor, ...}, {'col1': tensor, ...}]
    # We transform it into a dict of lists, then stack
    # This is significantly faster than row-by-row iteration
    inputs = torch.stack([
        torch.stack([item[col] for col in feature_columns]) 
        for item in batch
    ]).to(torch.float32)
    
    labels = torch.stack([item["Label"] for item in batch]).unsqueeze(1).to(torch.float32)
    
    return inputs, labels

# Initialize DataLoader
gen_train = DataLoader(train_dataset, batch_size=batch_size, collate_fn=fast_collate)

# 4. TEST: Print the first batch
print("\n--- Iterating over the first batch ---")
for i, (inputs, labels) in enumerate(gen_train):
    print(f"Batch index: {i}")
    print(f"Inputs Shape: {inputs.shape} (Expected: [{batch_size}, {len(feature_columns)}])")
    print(f"Labels Shape: {labels.shape} (Expected: [{batch_size}, 1])")
    
    print("\nFirst 3 rows of Inputs (Features):")
    print(inputs[:3])
    
    print("\nFirst 3 Labels:")
    print(labels[:3])
    
    # Break after first batch just to show it works
    break

# Cleanup
if os.path.exists(f"{input_file}.parquet"):
    os.remove(f"{input_file}.parquet")