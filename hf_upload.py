#!/usr/bin/env python3
"""
Script to upload LogicIF benchmark files to Hugging Face dataset repository.
Ensures code_output field is formatted as JSON string for type consistency.
"""

import json
import os
from datasets import Dataset
from huggingface_hub import HfApi

def process_jsonl_file(file_path):
    """
    Process a JSONL file and ensure code_output field is a JSON string.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of processed records
    """
    processed_records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            record = json.loads(line)

            del record['description']
            
            if 'input' in record and not isinstance(record['input'], str):
                record['input'] = json.dumps(record['input'])

            # Convert code_output to JSON string if it's not already
            if 'code_output' in record and not isinstance(record['code_output'], str):
                record['code_output'] = json.dumps(record['code_output'])
            
            processed_records.append(record)
    
    return processed_records

def upload_to_hf(repo_id="billmianz/LogicIFEval"):
    """
    Upload the benchmark files to Hugging Face dataset repository.
    
    Args:
        repo_id (str): Hugging Face repository ID
    """
    # Define file paths
    mini_file = "benchmark/logic-if-eval-mini.jsonl"
    full_file = "benchmark/logic-if-eval.jsonl"
    
    # Check if files exist
    if not os.path.exists(mini_file):
        raise FileNotFoundError(f"File not found: {mini_file}")
    if not os.path.exists(full_file):
        raise FileNotFoundError(f"File not found: {full_file}")
    
    print("Processing mini dataset...")
    mini_data = process_jsonl_file(mini_file)
    print(f"Processed {len(mini_data)} records from mini dataset")
    
    print("Processing full dataset...")
    full_data = process_jsonl_file(full_file)
    print(f"Processed {len(full_data)} records from full dataset")
    
    # Create datasets
    mini_dataset = Dataset.from_list(mini_data)
    full_dataset = Dataset.from_list(full_data)
    
    print("Creating dataset splits...")
    dataset_dict = {
        "mini": mini_dataset,
        "full": full_dataset
    }
    
    # Upload to Hugging Face
    print(f"Uploading to {repo_id}...")
    
    # Create dataset splits and push to hub
    from datasets import DatasetDict
    dataset = DatasetDict(dataset_dict)
    
    dataset.push_to_hub(
        repo_id=repo_id,
        commit_message="Upload LogicIF evaluation benchmark datasets"
    )
    
    print(f"Successfully uploaded datasets to {repo_id}")
    print("Dataset splits:")
    print(f"  - mini: {len(mini_data)} examples")
    print(f"  - full: {len(full_data)} examples")

def main():
    """Main function to run the upload process."""
    try:
        upload_to_hf()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 