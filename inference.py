"""
LLM Inference module for LogicIF framework.

This module provides functionality to run VLLM inference on benchmark JSONL files
and add LLM responses to each test case entry.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



def extract_model_name_for_filename(model: str) -> str:
    """Extract a clean model name from full path for use in filenames.
    
    Converts paths like:
    - ${HF_HOME}/Qwen/Qwen3-8B -> Qwen/Qwen3-8B
    - /path/to/models/meta-llama/Llama-2-7b-hf -> meta-llama/Llama-2-7b-hf
    - gpt-4 -> gpt-4
    """
    # Handle environment variable expansion
    expanded_model = os.path.expandvars(model)
    
    # Split path into components
    path_parts = expanded_model.split('/')
    
    # If it's already a simple model name (no slashes), return as-is
    if len(path_parts) <= 1:
        return model
    
    # For paths, take the last two non-empty components
    # This handles cases like /path/to/models/org/model-name
    non_empty_parts = [part for part in path_parts if part.strip()]
    if len(non_empty_parts) >= 2:
        return f"{non_empty_parts[-2]}/{non_empty_parts[-1]}"
    elif len(non_empty_parts) == 1:
        return non_empty_parts[-1]
    else:
        return model


def get_llm_response_filename(model: str, run_name: Optional[str] = None) -> str:
    """Generate a filename for LLM responses based on model name and run name."""
    # Extract clean model name for filename
    clean_model = extract_model_name_for_filename(model)
    safe_model = re.sub(r'[^A-Za-z0-9]+', '_', clean_model)
    
    # Add run name if provided
    run_suffix = f"_{run_name}" if run_name else ""
    return f"llm_responses_{safe_model}{run_suffix}.jsonl"


class VLLMInference:
    """VLLM inference engine for processing benchmark files."""
    
    def __init__(self, model_id: str, max_tokens: int = 2048, gpu_memory_utilization: float = 0.8):
        """
        Initialize VLLM inference engine.
        
        Args:
            model_id: HuggingFace model ID or path
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Get GPU number from CUDA_VISIBLE_DEVICES environment variable
        self.gpu_num = 1  # Default to 1 GPU
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices:
            # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
            # Format can be "0", "0,1", "0,1,2", etc.
            self.gpu_num = len(cuda_devices.split(','))
        
        print(f"INFO: Initializing VLLM with model: {model_id}")
        print(f"INFO: Using {self.gpu_num} GPU(s), max_tokens: {max_tokens}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Initialize VLLM model
        self.sampling_params = SamplingParams(max_tokens=max_tokens)
        self.llm = LLM(
            model=model_id,
            dtype='bfloat16',
            tensor_parallel_size=self.gpu_num,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
    
    def format_prompt(self, instruction: str) -> str:
        """Format instruction into a chat template for the model."""
        conv = [{"role": "user", "content": instruction}]
        
        # Handle special model cases
        if 'qwen3' in self.model_id.lower() or ('gemini' in self.model_id.lower() and '2.5' in self.model_id):
            return self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            return self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def process_benchmark_file(self, input_file: str, output_file: str, run_name: Optional[str] = None, 
                             overwrite: bool = False) -> None:
        """
        Process a benchmark JSONL file and add LLM responses.
        
        Args:
            input_file: Path to input benchmark JSONL file
            output_file: Path to output file with LLM responses
            run_name: Optional run name for tracking different experiments
            overwrite: Whether to overwrite existing output file
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if output_path.exists() and not overwrite:
            print(f"WARNING: Output file already exists: {output_file}. Use overwrite=True to replace.")
            return
        
        print(f"INFO: Loading benchmark data from: {input_file}")
        
        # Load benchmark entries
        benchmark_entries = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    benchmark_entries.append(json.loads(line))
        
        print(f"INFO: Loaded {len(benchmark_entries)} benchmark entries")
        
        # Prepare prompts
        print("INFO: Preparing prompts...")
        prompts = []
        for entry in benchmark_entries:
            instruction = entry.get('instruction', '')
            if not instruction:
                print(f"WARNING: Empty instruction found in entry {entry.get('task_id', 'unknown')}")
                prompts.append("")
            else:
                formatted_prompt = self.format_prompt(instruction)
                prompts.append(formatted_prompt)
        
        # Generate responses
        print(f"INFO: Generating responses with {self.model_id}...")
        responses = []
        batch_size = 32  # Process in batches to avoid memory issues
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = self.generate_responses(batch_prompts)
            responses.extend(batch_responses)
        
        # Add responses to benchmark entries
        print("INFO: Adding responses to benchmark entries...")
        enhanced_entries = []
        for entry, response in zip(benchmark_entries, responses):
            enhanced_entry = entry.copy()
            enhanced_entry['llm_response'] = response
            enhanced_entry['model_used'] = self.model_id
            if run_name:
                enhanced_entry['run_name'] = run_name
            enhanced_entries.append(enhanced_entry)
        
        # Save results
        print(f"INFO: Saving enhanced benchmark to: {output_file}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in enhanced_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"SUCCESS: Successfully processed {len(enhanced_entries)} entries")
        
        # Print statistics
        non_empty_responses = sum(1 for r in responses if r.strip())
        success_rate = (non_empty_responses / len(responses)) * 100 if responses else 0
        print(f"INFO: Success rate: {success_rate:.1f}% ({non_empty_responses}/{len(responses)} non-empty responses)")
    
    def __del__(self):
        """Clean up VLLM model when object is destroyed."""
        if hasattr(self, 'llm'):
            del self.llm


def run_vllm_inference(input_file: str, output_file: str, model_id: str, 
                      max_tokens: int = 4096, run_name: Optional[str] = None,
                      overwrite: bool = False, gpu_memory_utilization: float = 0.8) -> None:
    """
    Convenience function to run VLLM inference on a benchmark file.
    
    Args:
        input_file: Path to input benchmark JSONL file
        output_file: Path to output file with LLM responses  
        model_id: HuggingFace model ID or path
        max_tokens: Maximum tokens to generate
        run_name: Optional run name for tracking experiments
        overwrite: Whether to overwrite existing output file
        gpu_memory_utilization: GPU memory utilization ratio
    """
    inference_engine = VLLMInference(
        model_id=model_id,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    inference_engine.process_benchmark_file(
        input_file=input_file,
        output_file=output_file,
        run_name=run_name,
        overwrite=overwrite
    )


def main():
    """Main function for console script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VLLM inference on LogicIF benchmark files")
    parser.add_argument("--input", required=True, help="Input benchmark JSONL file")
    parser.add_argument("--output", required=True, help="Output file with LLM responses")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--run-name", help="Optional run name for tracking experiments")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, 
                       help="GPU memory utilization ratio")
    
    args = parser.parse_args()
    
    run_vllm_inference(
        input_file=args.input,
        output_file=args.output,
        model_id=args.model,
        max_tokens=args.max_tokens,
        run_name=args.run_name,
        overwrite=args.overwrite,
        gpu_memory_utilization=args.gpu_memory_utilization
    )


if __name__ == "__main__":
    main() 