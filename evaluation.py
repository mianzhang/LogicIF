"""
Evaluation module for LogicIF framework.

This module provides functionality to extract LLM results and compare outputs
for benchmark evaluation.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from .utils import openai_inference, load_api_keys
from .prompts import prompt_extract_output


def analyze_output_format(output_value):
    """
    Analyze the format/structure of an output value and return a string description.
    
    Args:
        output_value: The output value to analyze
        
    Returns:
        String description of the format
    """
    if output_value is None:
        return "None"
    elif isinstance(output_value, bool):
        return "bool"
    elif isinstance(output_value, int):
        return "int"
    elif isinstance(output_value, float):
        return "float"
    elif isinstance(output_value, str):
        return "str"
    elif isinstance(output_value, list):
        if len(output_value) == 0:
            return "list[]"
        
        # Check if all elements are the same type
        first_type = type(output_value[0])
        if all(isinstance(item, first_type) for item in output_value):
            if first_type == int:
                return "list[int]"
            elif first_type == float:
                return "list[float]"
            elif first_type == str:
                return "list[str]"
            elif first_type == bool:
                return "list[bool]"
            elif first_type == list:
                # Nested list - analyze the structure of first element
                inner_format = analyze_output_format(output_value[0])
                return f"list[{inner_format}]"
            else:
                return f"list[{first_type.__name__}]"
        else:
            # Mixed types - show the types of first few elements
            types_seen = [type(item).__name__ for item in output_value[:3]]
            if len(output_value) > 3:
                return f"list[mixed: {', '.join(types_seen)}, ...]"
            else:
                return f"list[mixed: {', '.join(types_seen)}]"
    
    elif isinstance(output_value, tuple):
        if len(output_value) == 0:
            return "tuple()"
        
        # Analyze each element of the tuple
        element_formats = []
        for item in output_value:
            element_formats.append(analyze_output_format(item))
        
        return f"tuple({', '.join(element_formats)})"
    
    elif isinstance(output_value, dict):
        if len(output_value) == 0:
            return "dict{}"
        
        # Analyze a few key-value pairs
        sample_items = list(output_value.items())[:3]
        key_types = set(type(k).__name__ for k, v in sample_items)
        value_types = set(type(v).__name__ for k, v in sample_items)
        
        if len(key_types) == 1 and len(value_types) == 1:
            return f"dict[{list(key_types)[0]}: {list(value_types)[0]}]"
        else:
            return f"dict[mixed keys/values]"
    
    elif isinstance(output_value, set):
        if len(output_value) == 0:
            return "set()"
        
        # Check element types
        first_item = next(iter(output_value))
        first_type = type(first_item)
        if all(isinstance(item, first_type) for item in output_value):
            return f"set[{first_type.__name__}]"
        else:
            return "set[mixed]"
    
    else:
        # Unknown/custom type
        return f"{type(output_value).__name__}"


def extract_output(desc: str, response: str, stats_keys: List[str], extract_model: str, 
                  output_format: str = "unknown") -> Tuple[Any, Dict, str, str]:
    """
    Extract structured output and statistics from LLM response using GPT-4.
    
    Args:
        desc: Algorithm description
        response: LLM response to extract from
        stats_keys: Expected statistics keys
        extract_model: Model to use for extraction (e.g., 'gpt-4o-mini')
        output_format: Expected output format
        
    Returns:
        Tuple of (output, stats, response, extract_response)
    """
    def gen_forward():
        try:
            conv = [
                {"role": "user", "content": prompt_extract_output.format(
                    algo_description=desc, 
                    llm_response=response, 
                    stats_keys=stats_keys,
                    output_format=output_format
                )},
            ]
            extract_response = openai_inference([conv], model=extract_model, return_json=True)[0]
            
            # Check if the extraction response is "[ERROR]"
            if extract_response == "[ERROR]":
                return {}, "[ERROR]", response
                
            json_obj = json.loads(extract_response)
            return json_obj, extract_response, response
        except json.JSONDecodeError as e:
            print(f"    JSON decode error in extract_output: {e}", flush=True)
            if 'extract_response' in locals():
                print(f"    Raw response: {extract_response[:200]}...", flush=True)
            # Return empty dict to trigger retry
            return {}, extract_response if 'extract_response' in locals() else "", response
        except Exception as e:
            print(f"    Unexpected error in extract_output: {e}", flush=True)
            return {}, "", response

    max_retries = 3
    base_delay = 1.0
    retry_count = 0
    
    while retry_count < max_retries:
        json_obj, extract_response, response = gen_forward()
        
        # Check if we got "[ERROR]" response and retry
        if extract_response == "[ERROR]":
            if retry_count < max_retries - 1:
                delay = base_delay * (2 ** retry_count)  # Exponential backoff
                time.sleep(delay)
                retry_count += 1
                continue
            else:
                # Final attempt failed with "[ERROR]"
                return None, {}, response, "[ERROR]"
        
        # Check if we have required fields
        if 'output' in json_obj and 'stats' in json_obj:
            output = json_obj.get('output', None)
            stats = json_obj.get('stats', {})
            return output, stats, response, extract_response
        
        # Missing required fields, retry
        if retry_count < max_retries - 1:
            retry_count += 1
            delay = base_delay * (2 ** (retry_count - 1))
            time.sleep(delay)
            continue
        else:
            break
    
    # All retries failed, return what we have
    output = json_obj.get('output', None) if json_obj else None
    stats = json_obj.get('stats', {}) if json_obj else {}
    
    return output, stats, response, extract_response


def has_error_responses_in_extractions(llm_output_path: Path) -> bool:
    """Check if LLM output file contains '[ERROR]' extraction responses."""
    try:
        with open(llm_output_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("extract_response") == "[ERROR]":
                        return True
        return False
    except Exception:
        return False


class LogicIFEvaluator:
    """Evaluator for LogicIF benchmark results."""
    
    def __init__(self):
        pass
    
    def extract_llm_results(self, input_file: str, output_file: Optional[str] = None,
                          extract_model: str = "gpt-4o-mini", 
                          overwrite: bool = False) -> None:
        """
        Extract structured results from LLM responses in inference output files.
        
        Args:
            input_file: Path to inference output JSONL file (with llm_response field)
            output_file: Path to output file with extracted results (optional)
            extract_model: Model to use for extraction
            overwrite: Whether to overwrite existing output files
        """
        print(f"\n=== Extracting LLM Results ===")
        print(f"Input file: {input_file}")
        print(f"Extract model: {extract_model}")
        
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_file}")
            print("Program terminated due to missing input file.")
            exit(1)
        
        # Load inference output entries
        inference_entries = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    inference_entries.append(json.loads(line))
        
        print(f"Loaded {len(inference_entries)} inference entries")
        
        # Prepare output data structure
        enhanced_entries = []
        successful_extractions = 0
        total_extractions = 0
        
        for entry in tqdm(inference_entries, desc="Extracting LLM results"):
            task_id = entry.get('task_id')
            test_case_id = entry.get('test_case_id', 0)
            llm_response = entry.get("llm_response", "")
            
            # Get expected output structure from code_output
            code_output = entry.get("code_output", {})
            expected_stats = code_output.get("stats", {})
            stats_keys = list(expected_stats.keys()) if expected_stats else []
            
            # Analyze expected output format
            expected_output = code_output.get("output")
            output_format = analyze_output_format(expected_output) if expected_output is not None else "unknown"
            
            # Require description to exist; raise error and exit if missing
            if "description" not in entry or not entry["description"].strip():
                print(f"ERROR: Missing description for task_id={task_id}, test_case_id={test_case_id}")
                print("Program terminated due to missing description field.")
                exit(1)
            description = entry["description"]
            
            total_extractions += 1
            
            if not llm_response.strip():
                result_dict = {
                    "output": None,
                    "stats": {},
                    "success": False,
                    "error": "LLM response is empty",
                    "extract_response": ""
                }
            else:
                try:
                    output, stats, response, extract_response = extract_output(
                        description, llm_response, stats_keys, extract_model, output_format
                    )
                    
                    # Check if extraction returned "[ERROR]" after all retries
                    success = extract_response != "[ERROR]"
                    if success:
                        successful_extractions += 1
                    
                    result_dict = {
                        "output": output,
                        "stats": stats,
                        "success": success,
                        "extract_response": extract_response
                    }
                    
                except Exception as e:
                    result_dict = {
                        "output": None,
                        "stats": {},
                        "success": False,
                        "error": str(e),
                        "extract_response": ""
                    }
            
            # Create enhanced entry for output file
            enhanced_entry = entry.copy()
            enhanced_entry["llm_output"] = result_dict
            enhanced_entries.append(enhanced_entry)
        
        # Save enhanced entries to output file
        if output_file:
            output_path = Path(output_file)
        else:
            # Generate default output filename
            input_path = Path(input_file)
            output_path = input_path.parent / f"{input_path.stem}_with_extractions.jsonl"
        
        if output_path.exists() and not overwrite:
            print(f"Output file already exists: {output_path}. Use overwrite=True to replace.")
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in enhanced_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"Enhanced benchmark with extractions saved to: {output_path}")
        
        # Print summary
        success_rate = (successful_extractions / total_extractions) * 100 if total_extractions > 0 else 0
        print(f"\n=== Extraction Summary ===")
        print(f"Extraction success rate: {success_rate:.1f}% ({successful_extractions}/{total_extractions})")
        
        error_count = total_extractions - successful_extractions
        if error_count > 0:
            print(f"{error_count} extractions failed or returned '[ERROR]'")
        else:
            print(f"All extractions completed successfully")
    
    def compare_outputs(self, result_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare LLM outputs with expected code outputs using extraction results file.
        
        Args:
            result_file: Path to result file with extracted results (optional, defaults to extracted_results.jsonl)
            
        Returns:
            Dictionary with comparison results and statistics
        """
        print(f"\n=== Comparing Outputs ===")
        extraction_file = Path(result_file)
        
        if not extraction_file.exists():
            print(f"ERROR: Extraction file not found: {extraction_file}")
            print("Program terminated due to missing extraction file.")
            exit(1)
        
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
        
        # Check if entries have difficulty field
        has_difficulty = any('difficulty' in entry for entry in extraction_entries)
        if has_difficulty:
            print("Difficulty field detected - will analyze by difficulty levels")
        
        # Prepare results
        question_level_results = {
            "output_correct": 0,
            "stats_correct": 0, 
            "both_correct": 0,
            "total_questions": 0
        }
        
        comparison_results = []
        difficulty_stats = {} if has_difficulty else None
        
        for task_id, entries in tqdm(functions_data.items(), desc="Comparing outputs"):
            # Compare outputs for this function
            output_matches = 0
            stats_matches = 0
            both_matches = 0
            total_cases = len(entries)
            successful_cases = 0
            
            # Get difficulty from first entry if available
            difficulty = entries[0].get("difficulty", "unknown") if has_difficulty else None
            
            for entry in entries:
                code_output = entry.get("code_output", {})
                llm_output = entry.get("llm_output", {})
                
                successful_cases += 1
                
                # Compare outputs and stats
                expected_output = code_output.get("output")
                expected_stats = code_output.get("stats", {})
                actual_output = llm_output.get("output")
                actual_stats = llm_output.get("stats", {})
                
                output_match = expected_output == actual_output
                stats_match = expected_stats == actual_stats
                both_match = output_match and stats_match
                
                if output_match:
                    output_matches += 1
                if stats_match:
                    stats_matches += 1
                if both_match:
                    both_matches += 1
            
            # Question-level accuracy: all test cases must pass for the question to be correct
            question_level_results["total_questions"] += 1
            if successful_cases > 0:  # Only count if there were successful cases
                if output_matches == successful_cases:
                    question_level_results["output_correct"] += 1
                if stats_matches == successful_cases:
                    question_level_results["stats_correct"] += 1
                if both_matches == successful_cases:
                    question_level_results["both_correct"] += 1
            
            # Create result entry
            result_entry = {
                "task_id": task_id,
                "total_test_cases": total_cases,
                "successful_cases": successful_cases,
                "output_matches": output_matches,
                "stats_matches": stats_matches,
                "both_matches": both_matches,
                "output_correct": output_matches == successful_cases if successful_cases > 0 else False,
                "stats_correct": stats_matches == successful_cases if successful_cases > 0 else False,
                "both_correct": both_matches == successful_cases if successful_cases > 0 else False
            }
            
            if has_difficulty:
                result_entry["difficulty"] = difficulty
            
            comparison_results.append(result_entry)
            print(f"  {task_id}: {both_matches}/{successful_cases} both matches")
            
            # Update difficulty stats if available
            if has_difficulty and difficulty_stats is not None:
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {
                        "count": 0,
                        "total_cases": 0,
                        "successful_cases": 0,
                        "output_matches": 0,
                        "stats_matches": 0,
                        "both_matches": 0,
                        "output_correct": 0,
                        "stats_correct": 0,
                        "both_correct": 0
                    }
                
                stats = difficulty_stats[difficulty]
                stats["count"] += 1
                stats["total_cases"] += result_entry["total_test_cases"]
                stats["successful_cases"] += result_entry["successful_cases"]
                stats["output_matches"] += result_entry["output_matches"]
                stats["stats_matches"] += result_entry["stats_matches"]
                stats["both_matches"] += result_entry["both_matches"]
                if result_entry["output_correct"]:
                    stats["output_correct"] += 1
                if result_entry["stats_correct"]:
                    stats["stats_correct"] += 1
                if result_entry["both_correct"]:
                    stats["both_correct"] += 1
        
        # Calculate and print summary
        if comparison_results:
            total_problems = len(comparison_results)
            total_test_cases = sum(r["total_test_cases"] for r in comparison_results)
            total_successful_cases = sum(r["successful_cases"] for r in comparison_results)
            total_output_matches = sum(r["output_matches"] for r in comparison_results)
            total_stats_matches = sum(r["stats_matches"] for r in comparison_results)
            total_both_matches = sum(r["both_matches"] for r in comparison_results)
            
            # Question-level accuracies
            output_accuracy = (question_level_results["output_correct"] / question_level_results["total_questions"]) * 100
            stats_accuracy = (question_level_results["stats_correct"] / question_level_results["total_questions"]) * 100
            both_accuracy = (question_level_results["both_correct"] / question_level_results["total_questions"]) * 100
            
            print(f"\n=== Comparison Summary ===")
            print(f"Total problems analyzed: {total_problems}")
            print(f"Total test cases: {total_test_cases}")
            print(f"Total successful cases: {total_successful_cases}")
            print(f"Total output matches: {total_output_matches}")
            print(f"Total stats matches: {total_stats_matches}")
            print(f"Total both matches: {total_both_matches}")
            
            print(f"\n=== Accuracy ===")
            print(f"Output accuracy: {question_level_results['output_correct']}/{question_level_results['total_questions']} ({output_accuracy:.1f}%)")
            print(f"Stats accuracy: {question_level_results['stats_correct']}/{question_level_results['total_questions']} ({stats_accuracy:.1f}%)")
            print(f"Both accuracy: {question_level_results['both_correct']}/{question_level_results['total_questions']} ({both_accuracy:.1f}%)")
            
            # Show breakdown by difficulty if available
            if has_difficulty and difficulty_stats:
                print(f"\nBreakdown by difficulty:")
                for difficulty_level, stats in sorted(difficulty_stats.items()):
                    if stats["count"] > 0:
                        output_q_rate = (stats["output_correct"] / stats["count"]) * 100
                        stats_q_rate = (stats["stats_correct"] / stats["count"]) * 100
                        both_q_rate = (stats["both_correct"] / stats["count"]) * 100
                        print(f"  {difficulty_level}: {stats['count']} problems")
                        print(f"    Output: {stats['output_correct']} ({output_q_rate:.1f}%)")
                        print(f"    Stats: {stats['stats_correct']} ({stats_q_rate:.1f}%)")
                        print(f"    Both: {stats['both_correct']} ({both_q_rate:.1f}%)")
        
        print("Output comparison completed!")
        
        result = {
            "total_problems": question_level_results["total_questions"],
            "question_level_accuracy": {
                "output": output_accuracy,
                "stats": stats_accuracy,
                "both": both_accuracy
            }
        }
        
        if has_difficulty and difficulty_stats:
            result["difficulty_breakdown"] = difficulty_stats
        
        return result


def main():
    """Main function for console script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LogicIF Evaluation Tools")
    parser.add_argument("--todo", required=True, choices=["extract_llm_results", "compare_outputs"],
                       help="Task to run")
    parser.add_argument("--input_file", help="Input JSONL file with LLM responses (for extract_llm_results)")
    parser.add_argument("--output_file", help="Output JSONL file with extracted results (for extract_llm_results)")
    parser.add_argument("--extract_model", default="gpt-4o-mini",
                       help="Model to use for extraction (required for extract_llm_results)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output files")
    parser.add_argument("--result_file", help="Path to result file with extracted results (for compare_outputs)")
    args = parser.parse_args()
    
    evaluator = LogicIFEvaluator()
    
    if args.todo == "extract_llm_results":
        if not args.input_file:
            parser.error("--input_file is required for extract_llm_results")
        if not args.output_file:
            parser.error("--output_file is required for extract_llm_results")

        load_api_keys()
        
        evaluator.extract_llm_results(
            input_file=args.input_file,
            output_file=args.output_file,
            extract_model=args.extract_model,
            overwrite=args.overwrite,
        )
    
    elif args.todo == "compare_outputs":
        if not args.result_file:
            parser.error("--result_file is required for compare_outputs")
        evaluator.compare_outputs(
            result_file=args.result_file
        )
    
    else:
        parser.error(f"Unknown todo task: {args.todo}. Use 'extract_llm_results' or 'compare_outputs'")


if __name__ == "__main__":
    main() 