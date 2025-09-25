"""
Evaluation module for LogicIF framework.

This module provides functionality to compare LLM outputs with expected outputs
for benchmark evaluation.
"""

import json
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm

import jsonparse


def eval_response_file(result_file: str) -> Dict[str, Any]:
    """
    Compare LLM outputs with expected code outputs using extraction results file.
    
    Args:
        result_file: Path to result file with extracted results
        
    Returns:
        Dictionary with comparison results and statistics
    """
    print(f"\n=== Comparing Outputs ===")
    extraction_file = Path(result_file)
    print(f"Using extraction file: {extraction_file}")
    
    # Load extraction results
    extraction_entries = []
    with open(extraction_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                extraction_entries.append(json.loads(line))
    
    print(f"Loaded {len(extraction_entries)} extraction entries")
    
    # Group entries by task_id (function)
    functions_data = {}
    for entry in extraction_entries:
        task_id = entry.get('task_id')
        if task_id not in functions_data:
            functions_data[task_id] = []
        functions_data[task_id].append(entry)
    
    print(f"Processing {len(functions_data)} functions")
    comparison_results = []
    for task_id, entries in tqdm(functions_data.items(), desc="Comparing outputs"):
        # Compare outputs for this function
        output_matches = 0
        stats_matches = 0
        both_matches = 0
        total_cases = len(entries)
        

        def check_llm_output_format(input):
            if not isinstance(input, dict):
                return False
            if "output" not in input:
                return False
            if "stats" not in input:
                return False
            return True

        for entry in entries:
            code_output = entry["code_output"]
            llm_output = jsonparse.extract_valid_json(entry["response"])
            
            # Compare outputs and stats
            expected_output = code_output["output"]
            expected_stats = code_output["stats"]
            actual_output = llm_output["output"] if llm_output is not None and check_llm_output_format(llm_output) else None
            actual_stats = llm_output["stats"] if llm_output is not None and check_llm_output_format(llm_output) else None
            
            output_match = expected_output == actual_output
            stats_match = expected_stats == actual_stats
            both_match = output_match and stats_match
            
            if output_match:
                output_matches += 1
            if stats_match:
                stats_matches += 1
            if both_match:
                both_matches += 1
        
        # Create result entry
        result_entry = {
            "task_id": task_id,
            "total_test_cases": total_cases,
            "output_matches": output_matches,
            "stats_matches": stats_matches,
            "both_matches": both_matches,
        }
        
        comparison_results.append(result_entry)
    
    # Calculate and print summary
    if comparison_results:
        total_test_cases = sum(r["total_test_cases"] for r in comparison_results)
        case_level_accuracy = sum(r["both_matches"] for r in comparison_results) / total_test_cases

        total_problems = len(comparison_results)
        question_level_accuracy = sum(r["both_matches"] == r["total_test_cases"] for r in comparison_results) / total_problems


        
    result = {
        "case_level_accuracy": case_level_accuracy,
        "question_level_accuracy": question_level_accuracy,
    }
    
    return result



if __name__ == "__main__":
    """Main function for console script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LogicIF Output Comparison Tool")
    parser.add_argument("--result_file", required=True, help="Path to result file with extracted results")
    args = parser.parse_args()
    
    result = eval_response_file(result_file=args.result_file)

    print(f"Case-level Accuracy: {result['case_level_accuracy']}")
    print(f"Question-level Accuracy: {result['question_level_accuracy']}")
    