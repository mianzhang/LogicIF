import os
import json
import multiprocessing
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

from .utils import openai_inference
from .prompts import (
    prompt_add_stat, prompt_get_desc, prompt_verify_description, 
    prompt_update_description, prompt_solve, prompt_evolve_function
)
from .ast_analyzer import analyze_function_ast


class TimeoutException(Exception):
    """Exception raised when function execution times out."""
    pass


class LogicIFGen:
    """
    LogicIF Generation Framework - Generate code instructions from functions and test cases.
    
    This framework allows users to:
    1. Load functions and test cases from JSONL files
    2. Add statistics tracking to functions
    3. Generate natural language descriptions
    4. Verify description completeness
    5. Execute functions on test cases to get outputs
    """
    
    def __init__(self, output_dir: str = "functions"):
        """
        Initialize the LogicIFGen framework.
        
        Args:
            output_dir: Directory to store generated files and results
        """
        self.output_dir = Path(output_dir)
        self.functions = []  # List of (name, function_string) tuples
        self.output_dir.mkdir(exist_ok=True)
        
    def load_functions_from_jsonl(self, jsonl_path: str, num_functions: Optional[int] = None) -> None:
        """
        Load functions from a JSONL file.
        
        Expected JSONL format:
        Each line should contain:
        {
            "function": "def func_name(...):\n    ...",
            "task_id": "unique_task_identifier"
        }
        
        Args:
            jsonl_path: Path to the JSONL file containing functions
            num_functions: Maximum number of functions to load (None for all)
        """
        self.functions = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_functions is not None and len(self.functions) >= num_functions:
                    break
                data = json.loads(line)
                
                task_id = data['task_id']
                func_str = data["function"]
                self.functions.append((task_id, func_str))
        
        print(f"Loaded {len(self.functions)} functions from {jsonl_path}")
    
    def load_test_cases(self, test_cases_file: str) -> None:
        """
        Load test cases from a JSONL file and create JSONL files for each function.

        Expected JSONL format (each line is a separate JSON object):
        {"task_id": "task_id_1", "test_cases": [[input1], [input2], ...]}
        {"task_id": "task_id_2", "test_cases": [[input1], [input2], ...]}

        Args:
            test_cases_file: Path to the JSONL file containing test cases
        """
        print(f"\n=== Loading Test Cases ===")

        try:
            test_cases_data = {}
            
            # Read JSONL file
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        task_id = data['task_id']
                        test_cases = data['test_cases']
                        test_cases_data[task_id] = test_cases

            print(f"Loaded test cases for {len(test_cases_data)} functions from {test_cases_file}")

            # Track statistics
            functions_with_test_cases = 0
            functions_without_test_cases = 0
            total_test_cases = 0

            # Get list of loaded function names for validation
            loaded_function_names = [name for name, _ in self.functions]

            # Create test case files for each function
            for func_name, cases in test_cases_data.items():
                assert func_name in loaded_function_names, f"Function {func_name} not found in loaded functions"

                folder_path = self._create_function_folder(func_name)
                test_cases_file_path = folder_path / "test_cases.jsonl"

                # Always overwrite test cases
                with open(test_cases_file_path, 'w', encoding='utf-8') as f:
                    for case in cases:
                        # Convert the list format to the expected {"input": [...]} format
                        case_obj = {"input": case}
                        f.write(json.dumps(case_obj) + "\n")

                functions_with_test_cases += 1
                total_test_cases += len(cases)
                print(f"  SUCCESS: Created {len(cases)} test cases for {func_name}")

            # Check for functions without test cases
            for func_name, _ in self.functions:
                if func_name not in test_cases_data:
                    functions_without_test_cases += 1
                    print(f"  WARNING: No test cases found for loaded function '{func_name}'")

            # Print summary
            print(f"\n=== Test Cases Loading Summary ===")
            print(f"Functions with test cases: {functions_with_test_cases}")
            print(f"Functions without test cases: {functions_without_test_cases}")
            print(f"Total test cases created: {total_test_cases}")

            if functions_without_test_cases > 0:
                print(f"WARNING: {functions_without_test_cases} loaded functions are missing test cases")

        except FileNotFoundError:
            print(f"ERROR: Test cases file '{test_cases_file}' not found")
            raise
        except json.JSONDecodeError as e:
            print(f"ERROR: Error parsing JSONL in '{test_cases_file}': {e}")
            raise
        except Exception as e:
            print(f"ERROR: Error loading test cases: {e}")
            raise
    
    def add_stat(self, model: str = "gpt-4o", reasoning_effort: str = 'medium', overwrite: bool = False) -> None:
        """
        Add statistics tracking to all loaded functions using LLM.
        
        Args:
            model: LLM model to use for adding statistics
            reasoning_effort: Reasoning effort level for OpenAI reasoning models
            overwrite: If True, overwrite existing files; if False, skip existing files
        """
        print(f"\n=== Adding Statistics to Functions ===")
        
        for name, func_str in tqdm(self.functions, desc="Adding statistics"):
            print(f"Processing function: {name}")
            
            folder_path = self._create_function_folder(name)
            
            # Check if already processed
            if (folder_path / "function_with_stats.py").exists() and not overwrite:
                print(f"  Statistics already added for {name}, skipping...")
                continue
            
            # Add statistics using LLM
            try:
                enhanced_func, stats_keys = self._add_stat_to_function(
                    func_str, model, reasoning_effort
                )
                
                if enhanced_func and stats_keys:
                    # Save enhanced function
                    self._save_to_function_folder(name, "function_with_stats.py", enhanced_func)
                    
                    # Save statistics metadata
                    stats_info = {"stats_keys": stats_keys}
                    self._save_to_function_folder(
                        name, "stats_info.json", 
                        json.dumps(stats_info, indent=2)
                    )
                    
                    print(f"  SUCCESS: Added {len(stats_keys)} statistics: {stats_keys}")
                else:
                    print(f"  ERROR: Failed to add statistics")
                    
            except Exception as e:
                print(f"  Error processing {name}: {e}")
    
    def gen_desc(self, model: str = "gpt-4o", reasoning_effort: str = 'medium', overwrite: bool = False) -> None:
        """
        Generate natural language descriptions for all loaded functions.
        
        Args:
            model: LLM model to use for generating descriptions
            reasoning_effort: Reasoning effort level for OpenAI reasoning models
            overwrite: If True, overwrite existing files; if False, skip existing files
        """
        print(f"\n=== Generating Descriptions ===")
        
        for name, func_str in tqdm(self.functions, desc="Generating descriptions"):
            print(f"Processing function: {name}")
            
            folder_path = self._create_function_folder(name)
            
            # Check if already processed
            if (folder_path / "description.txt").exists() and not overwrite:
                print(f"  Description already exists for {name}, skipping...")
                continue
            
            # Determine which function to use (prefer evolved, fallback to original)
            try:
                function_file, _, file_description = self._get_function_and_stats_files(name)
                print(f"  {file_description}")
                with open(function_file, 'r') as f:
                    function_code = f.read().strip()
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                print("  Program terminated due to missing function files.")
                exit(1)
            
            try:
                description = self._generate_description(function_code, model, reasoning_effort)
                
                if description:
                    self._save_to_function_folder(name, "description.txt", description)
                    print(f"  SUCCESS: Generated description")
                else:
                    print(f"  ERROR: Failed to generate description")
                    
            except Exception as e:
                print(f"  Error processing {name}: {e}")
    
    def verify_desc(self, model: str = "gpt-4o", reasoning_effort: str = 'medium', max_turns: int = 3, overwrite: bool = False) -> None:
        """
        Verify and refine the completeness of generated descriptions using multi-turn verification.
        
        Args:
            model: LLM model to use for verification and updating
            reasoning_effort: Reasoning effort level for OpenAI reasoning models
            max_turns: Maximum number of verification/refinement turns
            overwrite: If True, overwrite existing files; if False, skip existing files
        """
        print(f"\n=== Multi-Turn Description Verification & Refinement ===")
        print(f"Max turns: {max_turns}")
        
        for name, func_str in tqdm(self.functions, desc="Verifying descriptions"):
            print(f"Processing function: {name}")
            
            folder_path = self._create_function_folder(name)
            description_file = folder_path / "description.txt"
            
            # Check if already completed successfully
            if (folder_path / "description_success.txt").exists() and not overwrite:
                print(f"  Description already successfully verified for {name}, skipping...")
                continue
            
            if not description_file.exists():
                print(f"  No description found for {name}, skipping...")
                continue
            
            # Determine which function to use (prefer evolved, fallback to original)
            try:
                function_code_file, _, file_description = self._get_function_and_stats_files(name)
                print(f"  {file_description}")
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                print("  Program terminated due to missing function files.")
                exit(1)
            
            try:
                # Read initial description
                with open(description_file, 'r') as f:
                    current_description = f.read()
                
                # Read function code for verification
                with open(function_code_file, 'r') as f:
                    function_code = f.read()
                
                if not function_code.strip() or not current_description.strip():
                    print(f"  Empty file content for {name}, skipping...")
                    continue
                
                # Multi-turn verification and updating process
                turn = 1
                verification_history = []
                
                while turn <= max_turns:
                    print(f"  Turn {turn}/{max_turns}: Verifying description...")
                    
                    # Verify current description
                    verification_result = self._verify_description(
                        function_code, current_description, model, reasoning_effort
                    )
                    
                    verification_history.append({
                        "turn": turn,
                        "verification_result": verification_result,
                        "description_used": current_description
                    })
                    
                    # Save turn-specific description
                    self._save_to_function_folder(name, f"description_turn_{turn}.txt", current_description)
                    
                    is_complete = verification_result.get("desc_is_complete", False)
                    coverage = verification_result.get("coverage_percentage", "unknown")
                    
                    if is_complete:
                        # Verification passed - save as final success description
                        self._save_to_function_folder(name, "description_success.txt", current_description)
                        
                        # Save verification history
                        self._save_to_function_folder(
                            name, "verification_history.json",
                            json.dumps(verification_history, indent=2)
                        )
                        
                        print(f"  SUCCESS: Description complete after {turn} turn(s) (coverage: {coverage}%)")
                        break
                    else:
                        print(f"  WARNING: Turn {turn}: Description incomplete (coverage: {coverage}%)")
                        
                        if turn < max_turns:
                            # Update description for next turn
                            print(f"  INFO: Refining description based on feedback...")
                            updated_description = self._update_description(
                                function_code, current_description, verification_result, model, reasoning_effort
                            )
                            
                            if updated_description and updated_description != current_description:
                                current_description = updated_description
                                print(f"  📝 Description updated for next turn")
                            else:
                                print(f"  WARNING: Description update failed or no changes made")
                        else:
                            # Max turns reached
                            print(f"  ERROR: Max turns ({max_turns}) reached, description still incomplete")
                            
                            # Save final verification history
                            self._save_to_function_folder(
                                name, "verification_history.json",
                                json.dumps(verification_history, indent=2)
                            )
                            
                            # Save final verification result
                            final_result = verification_result.copy()
                            final_result.update({
                                "final_turn": turn,
                                "turns_attempted": turn,
                                "status": "max_turns_reached"
                            })
                            self._save_to_function_folder(
                                name, "verification_result.json",
                                json.dumps(final_result, indent=2)
                            )
                    
                    turn += 1
                    
            except Exception as e:
                print(f"  Error processing {name}: {e}")
    
    def evolve_functions(self, model: str = "gpt-4o", reasoning_effort: str = 'medium', max_turns: int = 3, overwrite: bool = False) -> None:
        """
        Evolve functions to make them more logically complicated using multi-turn LLM evolution.
        
        This method takes existing function_with_stats.py files and evolves them to be more
        sophisticated while maintaining the same input signature and core functionality.
        The final evolved function is saved as function_final.py (keeping original unchanged).
        
        Args:
            model: LLM model to use for evolution
            reasoning_effort: Reasoning effort level for OpenAI reasoning models  
            max_turns: Maximum number of evolution turns to apply
            overwrite: If True, overwrite existing files; if False, skip existing files
        """
        print(f"\n=== Multi-Turn Function Evolution ===")
        print(f"Max evolution turns: {max_turns}")
        
        for name, func_str in tqdm(self.functions, desc="Evolving functions"):
            print(f"Processing function: {name}")
            
            folder_path = self._create_function_folder(name)
            function_with_stats_file = folder_path / "function_with_stats.py"
            function_final_file = folder_path / "function_final.py"
            
            # Check if function_with_stats.py exists as the starting point
            if not function_with_stats_file.exists():
                print(f"  No function_with_stats.py found for {name}, skipping...")
                continue
            
            # Check if already evolved successfully (unless overwrite is True)
            if function_final_file.exists() and not overwrite:
                print(f"  Function already evolved successfully for {name}, skipping...")
                continue
            
            try:
                # Read the current function with stats (starting point for evolution)
                with open(function_with_stats_file, 'r') as f:
                    current_function = f.read().strip()
                
                if not current_function:
                    print(f"  Empty function file for {name}, skipping...")
                    continue
                
                # Multi-turn evolution process
                evolved_function = current_function
                evolution_history = []
                
                for turn in range(1, max_turns + 1):
                    print(f"  Evolution turn {turn}/{max_turns}...")
                    
                    # Evolve the function
                    evolution_result = self._evolve_function(evolved_function, model, reasoning_effort)
                    
                    if not evolution_result:
                        print(f"  ERROR: Evolution failed at turn {turn}")
                        break
                    
                    new_function = evolution_result.get("evolved_function", "")
                    new_stats_keys = evolution_result.get("stats_keys", [])
                    evolution_desc = evolution_result.get("evolution_description", "")
                    
                    if not new_function or not new_stats_keys:
                        print(f"  ERROR: Invalid evolution result at turn {turn}")
                        break
                    
                    # Validate the evolved function
                    try:
                        compile(new_function, '<string>', 'exec')
                    except SyntaxError as e:
                        print(f"  ERROR: Syntax error in evolved function at turn {turn}: {e}")
                        break
                    
                    # Update for next turn
                    evolved_function = new_function
                    
                    # Record this turn's evolution
                    turn_info = {
                        "turn": turn,
                        "evolution_description": evolution_desc,
                        "stats_keys": new_stats_keys,
                        "function_length": len(new_function.split('\n'))
                    }
                    evolution_history.append(turn_info)
                    
                    # Save intermediate result
                    self._save_to_function_folder(name, f"evolved_turn_{turn}.py", new_function)
                    
                    print(f"  SUCCESS: Turn {turn} complete: {evolution_desc}")
                
                if evolution_history:
                    # Save final evolved function as function_final.py (keep original unchanged)
                    self._save_to_function_folder(name, "function_final.py", evolved_function)
                    
                    # Save final stats info as stats_final.json
                    final_stats_keys = evolution_history[-1]["stats_keys"]
                    final_stats_info = {"stats_keys": final_stats_keys}
                    self._save_to_function_folder(
                        name, "stats_final.json", 
                        json.dumps(final_stats_info, indent=2)
                    )
                    
                    # Save evolution history
                    self._save_to_function_folder(
                        name, "evolution_history.json",
                        json.dumps(evolution_history, indent=2)
                    )
                    
                    print(f"  SUCCESS: Evolution completed after {len(evolution_history)} turns")
                    print(f"  INFO: Original preserved: function_with_stats.py")
                    print(f"  INFO: Final evolved: function_final.py")
                    print(f"  INFO: Final stats: {final_stats_keys}")
                else:
                    print(f"  ERROR: No successful evolution turns completed")
                    
            except Exception as e:
                print(f"  Error evolving {name}: {e}")
    
    def get_code_output(self, timeout: int = 5) -> None:
        """
        Execute functions on test cases to get code outputs and statistics.

        Args:
            timeout: Timeout in seconds for each function execution
        """
        print(f"\n=== Getting Code Outputs ===")

        functions_with_errors = []
        functions_with_timeouts = []
        functions_passing_all_tests = []

        for name, func_str in tqdm(self.functions, desc="Getting code outputs"):
            print(f"Processing function: {name}")

            folder_path = self._create_function_folder(name)

            # Always overwrite code_outputs.jsonl
            try:
                # Load function (prefer evolved, fallback to original)
                try:
                    function_file, _, file_description = self._get_function_and_stats_files(name)
                    print(f"  {file_description}")
                    with open(function_file, 'r') as f:
                        enhanced_func_str = f.read()
                except FileNotFoundError as e:
                    print(f"  ERROR: {e}")
                    print("  Program terminated due to missing function files.")
                    exit(1)

                # Load test cases
                test_cases_file = folder_path / "test_cases.jsonl"
                if not test_cases_file.exists():
                    print(f"  No test cases found for {name}, skipping...")
                    continue

                with open(test_cases_file, 'r') as f:
                    test_cases = [json.loads(line) for line in f]

                # Execute function on test cases
                outputs, has_error, has_timeout = self._execute_function_on_test_cases(
                    enhanced_func_str, test_cases, timeout
                )

                # Filter out successful outputs only (skip error cases)
                successful_outputs = [out for out in outputs if out["success"]]
                
                # Save only successful outputs
                if successful_outputs:
                    with open(folder_path / "code_outputs.jsonl", 'w') as f:
                        for output in successful_outputs:
                            f.write(json.dumps(output) + "\n")
                else:
                    # Remove existing file if no successful outputs
                    code_outputs_file = folder_path / "code_outputs.jsonl"
                    if code_outputs_file.exists():
                        code_outputs_file.unlink()

                # Track results based on successful cases only
                total_cases = len(test_cases)
                successful_cases = len(successful_outputs)
                error_cases = sum(1 for out in outputs if not out["success"] and out.get("error") != "Timeout")
                timeout_cases = sum(1 for out in outputs if out.get("error") == "Timeout")

                if has_timeout:
                    functions_with_timeouts.append(name)
                    print(f"  WARNING: {timeout_cases} test cases timed out, {successful_cases}/{total_cases} cases saved")
                elif successful_cases == 0:
                    functions_with_errors.append(name)
                    print(f"  ERROR: No test cases passed, {error_cases} errors")
                elif successful_cases == total_cases:
                    functions_passing_all_tests.append(name)
                    print(f"  SUCCESS: All {successful_cases} test cases passed")
                else:
                    print(f"  INFO: {successful_cases}/{total_cases} test cases passed, {error_cases} errors skipped")

            except Exception as e:
                print(f"  Error processing {name}: {e}")
                functions_with_errors.append(name)

        # Print summary
        print(f"\n=== Summary ===")
        print(f"Functions with errors: {len(functions_with_errors)}")
        print(f"Functions with timeouts: {len(functions_with_timeouts)}")
        print(f"Functions passing all tests: {len(functions_passing_all_tests)}")

    def analyze_complexity(self) -> None:
        """
        Analyze the complexity of each function using AST analysis and calculate complexity scores.
        Saves the results in a difficulty_info.json file in each function folder (contains complexity score only).
        
        Args:
        """
        print(f"\n=== Analyzing Function Complexity ===")
        
        functions_analyzed = 0
        functions_skipped = 0
        complexity_scores = []
        
        for name, func_str in tqdm(self.functions, desc="Analyzing complexity"):
            print(f"Processing function: {name}")
            
            folder_path = self._create_function_folder(name)
            difficulty_info_file = folder_path / "difficulty_info.json"
            
            # Check if already analyzed
            if difficulty_info_file.exists():
                print(f"  Complexity already analyzed for {name}, skipping...")
                functions_skipped += 1
                continue
            
            # Check for function files (prefer evolved, fallback to original)
            try:
                function_code_file, _, file_description = self._get_function_and_stats_files(name)
                print(f"  {file_description}")
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                print("  Program terminated due to missing function files.")
                exit(1)
            
            try:
                # Load function code if using file
                if function_code_file:
                    with open(function_code_file, 'r', encoding='utf-8') as f:
                        function_code = f.read()
                
                if not function_code.strip():
                    print(f"  Empty function code for {name}, skipping...")
                    functions_skipped += 1
                    continue
                
                # Perform AST analysis
                print(f"  Analyzing AST structure...")
                ast_analysis = analyze_function_ast(function_code)
                
                if not ast_analysis.get('parsing_success', False):
                    print(f"  ERROR: AST parsing failed: {ast_analysis.get('parsing_error', 'Unknown error')}")
                    complexity_score = 0
                    parsing_error = ast_analysis.get('parsing_error', 'Unknown error')
                else:
                    # Only calculate complexity score, no difficulty determination
                    complexity_score = ast_analysis.get('code_complexity_score', 0)
                    parsing_error = None
                    print(f"  INFO: Complexity score: {complexity_score}")
                
                # Use function name as task_id (since it's now directly the task_id from JSONL)
                task_id = name
                
                # Create complexity info (without difficulty determination)
                complexity_info = {
                    "function_name": name,
                    "task_id": task_id,
                    "complexity_score": complexity_score,
                    "ast_analysis": ast_analysis
                }
                
                # Add parsing error if it occurred
                if parsing_error:
                    complexity_info["parsing_error"] = parsing_error
                
                # Save complexity info
                self._save_to_function_folder(
                    name, "difficulty_info.json",
                    json.dumps(complexity_info, indent=2)
                )
                
                # Track complexity scores for statistics
                if complexity_score > 0:
                    complexity_scores.append(complexity_score)
                functions_analyzed += 1
                
                print(f"  SUCCESS: Complexity score: {complexity_score}")
                
            except Exception as e:
                print(f"  ERROR: Error analyzing {name}: {e}")
                functions_skipped += 1
        
        # Print summary
        print(f"\n=== Complexity Analysis Summary ===")
        print(f"Functions analyzed: {functions_analyzed}")
        print(f"Functions skipped: {functions_skipped}")
        
        if complexity_scores:
            print(f"\nComplexity Score Statistics:")
            print(f"  Min score: {min(complexity_scores):.2f}")
            print(f"  Max score: {max(complexity_scores):.2f}")
            print(f"  Average score: {sum(complexity_scores)/len(complexity_scores):.2f}")
        
        print("SUCCESS: Complexity analysis completed!")
    
    def finalize_benchmark(self, output_file: str = None, difficulty_file: str = None) -> None:
        """
        Collect all processed results into a single JSONL benchmark file.
        
        Args:
            output_file: Path to output JSONL file. If None, uses default naming.
            difficulty_file: Path to external file with complexity scores. If None, uses difficulty_info.json from function folders.
        """
        print(f"\n=== Finalizing Benchmark ===")
        print("Collecting benchmark data into a JSONL file where each line represents a test case...")
        
        # Load complexity score information
        complexity_mapping = {}
        if difficulty_file:
            # Use external difficulty file if provided (for backward compatibility)
            try:
                with open(difficulty_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        difficulty_info = json.loads(line.strip())
                        function_name = difficulty_info.get("function_name")
                        complexity_score = difficulty_info.get("complexity_score", 0)
                        if function_name:
                            complexity_mapping[function_name] = complexity_score
                print(f"SUCCESS: Loaded complexity scores for {len(complexity_mapping)} functions from {difficulty_file}")
            except FileNotFoundError:
                print(f"WARNING: External file not found: {difficulty_file}. Will use complexity scores from function folders.")
            except Exception as e:
                print(f"WARNING: Error loading external file: {e}. Will use complexity scores from function folders.")
        else:
            # Load complexity scores from individual function folders
            print("INFO: Loading complexity scores from function folders...")
            for name, _ in self.functions:
                folder_path = self._create_function_folder(name)
                difficulty_info_file = folder_path / "difficulty_info.json"
                if difficulty_info_file.exists():
                    try:
                        with open(difficulty_info_file, 'r', encoding='utf-8') as f:
                            complexity_info = json.load(f)
                            complexity_mapping[name] = complexity_info.get("complexity_score", 0)
                    except Exception as e:
                        print(f"WARNING: Error reading complexity info for {name}: {e}")
                        complexity_mapping[name] = 0
                else:
                    complexity_mapping[name] = 0
            print(f"SUCCESS: Loaded complexity scores for {len(complexity_mapping)} functions from individual folders")
        
        benchmark_entries = []
        functions_processed = 0
        functions_skipped = 0
        
        for name, func_str in tqdm(self.functions, desc="Finalizing benchmark"):
            # Use function name as task_id (since it's now directly the task_id from JSONL)
            task_id = name
            
            print(f"Processing function: {name} (task_id: {task_id})")
            
            folder_path = self._create_function_folder(name)
            
            # Check for required files
            description_success_file = folder_path / "description_success.txt"
            code_outputs_file = folder_path / "code_outputs.jsonl"
            
            # Require description_success.txt to exist (verified description)
            if not description_success_file.exists():
                print(f"  WARNING: No description_success.txt found for {name}. Skipping due to unverified description.")
                functions_skipped += 1
                continue
            
            # Determine which function file to use (prefer evolved, fallback to original)
            try:
                final_function_file, _, file_description = self._get_function_and_stats_files(name)
                print(f"  {file_description}")
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                print("  Program terminated due to missing function files.")
                exit(1)
            
            # Check for code outputs file
            if not code_outputs_file.exists():
                print(f"  WARNING: No code_outputs.jsonl found for {name}. Skipping.")
                functions_skipped += 1
                continue
            
            final_description_file = description_success_file
            
            try:
                # Load required data
                with open(final_function_file, "r", encoding='utf-8') as f:
                    function_code = f.read().strip()
                
                with open(final_description_file, "r", encoding='utf-8') as f:
                    description = f.read().strip()
                
                with open(code_outputs_file, "r") as f:
                    code_outputs_data = [json.loads(line) for line in f]
                
                # Create one entry per test case
                for test_case_id, case in enumerate(code_outputs_data):
                    # Extract input and code output
                    test_input = case.get('input', [])
                    code_output = {
                        'output': case.get('output', ''),
                        'stats': case.get('stats', {})
                    }
                    
                    # Generate instruction by filling the prompt_solve template
                    instruction = prompt_solve.format(
                        algo_description=description,
                        input=test_input
                    )
                    
                    # Get complexity score for this function
                    complexity_score = complexity_mapping.get(name, 0)
                    
                    # Create benchmark entry
                    benchmark_entry = {
                        "task_id": task_id,
                        "test_case_id": test_case_id,
                        "input": test_input,
                        "code_output": code_output,
                        "function": function_code,
                        "description": description,
                        "instruction": instruction,
                        "complexity_score": complexity_score
                    }
                    
                    benchmark_entries.append(benchmark_entry)
                
                functions_processed += 1
                print(f"  SUCCESS: Collected {len(code_outputs_data)} test cases from {name}")
                    
            except Exception as e:
                print(f"  ERROR: Error processing {name}: {e}")
                functions_skipped += 1
                continue
        
        # Save the benchmark entries to output file
        if benchmark_entries:
            # Use specified output_file or create default filename
            if output_file:
                final_output_file = output_file
            else:
                final_output_file = f"benchmark_logicif.jsonl"
            
            with open(final_output_file, "w", encoding='utf-8') as f:
                for entry in benchmark_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            print(f"\n=== Benchmark Finalization Summary ===")
            print(f"Functions processed: {functions_processed}")
            print(f"Functions skipped: {functions_skipped}")
            print(f"Total test case entries created: {len(benchmark_entries)}")
            print(f"Output saved to: {final_output_file}")
            
            print("SUCCESS: Benchmark finalization completed successfully!")
        else:
            print("ERROR: No benchmark entries were created. Check your function folders.")
    
    def _create_function_folder(self, func_name: str) -> Path:
        """Create a folder for a function and return the path."""
        folder_path = self.output_dir / func_name
        folder_path.mkdir(exist_ok=True)
        return folder_path
    
    def _get_function_and_stats_files(self, func_name: str) -> Tuple[Path, Optional[Path], str]:
        """
        Determine which function and stats files to use based on availability.
        
        Priority order:
        1. function_final.py + stats_final.json (evolved versions)
        2. function_with_stats.py + stats_info.json (original versions)
        
        Args:
            func_name: Name of the function
            
        Returns:
            Tuple of (function_file_path, stats_file_path, description)
            - function_file_path: Path to the function file to use
            - stats_file_path: Path to the stats file to use (None if none found)  
            - description: String describing which files are being used
            
        Raises:
            FileNotFoundError: If no function files are found
        """
        folder_path = self._create_function_folder(func_name)
        
        # Check for evolved versions first
        function_final_file = folder_path / "function_final.py"
        stats_final_file = folder_path / "stats_final.json"
        
        if function_final_file.exists():
            return (
                function_final_file,
                stats_final_file if stats_final_file.exists() else None,
                "Using evolved function: function_final.py"
            )
        
        # Fall back to original versions
        function_with_stats_file = folder_path / "function_with_stats.py"
        stats_info_file = folder_path / "stats_info.json"
        
        if function_with_stats_file.exists():
            return (
                function_with_stats_file,
                stats_info_file if stats_info_file.exists() else None,
                "Using original function: function_with_stats.py"
            )
        
        # No function files found - raise error
        raise FileNotFoundError(f"No function files found for '{func_name}'. Expected either function_final.py or function_with_stats.py in {folder_path}")
    
    def _save_to_function_folder(self, func_name: str, filename: str, content: str) -> None:
        """Save content to a file in the function's folder."""
        folder_path = self._create_function_folder(func_name)
        with open(folder_path / filename, 'w') as f:
            f.write(content)
    
    def _add_stat_to_function(self, func_str: str, model: str, reasoning_effort: str) -> Tuple[str, List[str]]:
        """Add statistics to a function using LLM."""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            conv = [
                {"role": "user", "content": prompt_add_stat.format(function=func_str)},
            ]
            response = openai_inference([conv], model=model, return_json=True, reasoning_effort=reasoning_effort)[0]
            
            if response == '[ERROR]':
                retry_count += 1
                continue
                
            try:
                json_obj = json.loads(response)
            except json.JSONDecodeError:
                retry_count += 1
                continue
            
            # Validate response
            if 'function' not in json_obj:
                retry_count += 1
                continue
                
            generated_function = json_obj['function']
            if not generated_function or not generated_function.strip():
                retry_count += 1
                continue
            
            stats_keys = json_obj.get('stats_keys', [])
            if not (1 <= len(stats_keys) <= 3):
                retry_count += 1
                continue
            
            # Validate syntax
            try:
                compile(generated_function, '<string>', 'exec')
                return (generated_function, stats_keys)
            except SyntaxError:
                retry_count += 1
                continue
        
        # All retries failed
        return ("", [])
    
    def _generate_description(self, func_str: str, model: str, reasoning_effort: str) -> str:
        """Generate a natural language description for a function."""
        conv = [
            {"role": "user", "content": prompt_get_desc.format(function=func_str)},
        ]
        response = openai_inference([conv], model=model, return_json=True, reasoning_effort=reasoning_effort)[0]
        
        if response == '[ERROR]':
            return ""
            
        try:
            json_obj = json.loads(response)
            # Convert sections into a single paragraph
            desc_paragraph = f"INPUTS: {json_obj['inputs']} \n\nLOGICS: {json_obj['logics']} \n\nOUTPUTS: {json_obj['outputs']}"
            return desc_paragraph
        except (json.JSONDecodeError, KeyError):
            return ""
    
    def _verify_description(self, function_code: str, description: str, model: str, reasoning_effort: str) -> Dict[str, Any]:
        """Verify if a description accurately covers the function code."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                conv = [
                    {"role": "user", "content": prompt_verify_description.format(
                        function_code=function_code, 
                        description=description
                    )},
                ]
                response = openai_inference([conv], model=model, return_json=True, reasoning_effort=reasoning_effort)[0]
                
                if response == '[ERROR]':
                    retry_count += 1
                    continue
                    
                json_obj = json.loads(response)
                
                # Validate required fields
                required_fields = ["desc_is_complete", "reasoning", "missing_aspects", "coverage_percentage"]
                if all(field in json_obj for field in required_fields):
                    # Ensure desc_is_complete is boolean
                    if isinstance(json_obj["desc_is_complete"], str):
                        json_obj["desc_is_complete"] = json_obj["desc_is_complete"].lower() in ['true', 'yes', '1']
                    return json_obj
                else:
                    retry_count += 1
                    
            except Exception as e:
                retry_count += 1
        
        # Fallback if all retries failed
        return {
            "desc_is_complete": False,
            "reasoning": "Verification failed due to LLM response issues",
            "missing_aspects": ["Verification process failed"],
            "coverage_percentage": "unknown"
        }
    
    def _update_description(self, function_code: str, current_description: str, verification_result: Dict[str, Any], model: str, reasoning_effort: str) -> str:
        """
        Use LLM to update a description based on verification feedback.
        
        Args:
            function_code: String containing the function code
            current_description: String containing the current description
            verification_result: Dictionary containing verification feedback
            model: Model to use for updating
            reasoning_effort: Reasoning effort level for OpenAI reasoning models
            
        Returns:
            String with updated description in the same format as original
        """
        max_retries = 3
        retry_count = 0
        
        missing_aspects = verification_result.get("missing_aspects", [])
        verification_reasoning = verification_result.get("reasoning", "")
        
        while retry_count < max_retries:
            try:
                conv = [
                    {"role": "user", "content": prompt_update_description.format(
                        function_code=function_code,
                        current_description=current_description,
                        missing_aspects=", ".join(missing_aspects),
                        verification_reasoning=verification_reasoning
                    )},
                ]
                response = openai_inference([conv], model=model, return_json=True, reasoning_effort=reasoning_effort)[0]
                
                if response == '[ERROR]':
                    retry_count += 1
                    continue
                    
                json_obj = json.loads(response)
                
                # Validate that all required sections are present
                required_sections = ["inputs", "logics", "outputs"]
                if all(section in json_obj and json_obj[section].strip() for section in required_sections):
                    # Convert the three sections back to the same format as original descriptions
                    updated_description = f"INPUTS: {json_obj['inputs']} \n\nLOGICS: {json_obj['logics']} \n\nOUTPUTS: {json_obj['outputs']}"
                    return updated_description
                else:
                    retry_count += 1
                    missing_sections = [s for s in required_sections if s not in json_obj or not json_obj[s].strip()]
                    print(f"    Missing or empty sections in JSON response: {missing_sections}. Retrying... ({retry_count}/{max_retries})")
                    
            except json.JSONDecodeError as e:
                retry_count += 1
                print(f"    JSON decode error in description update (attempt {retry_count}): {e}")
            except Exception as e:
                retry_count += 1
                print(f"    Error in description update (attempt {retry_count}): {e}")
        
        # Fallback if all retries failed
        print(f"    Warning: Description update failed, using original description")
        return current_description
    
    def _evolve_function(self, function_code: str, model: str, reasoning_effort: str) -> Dict[str, Any]:
        """
        Use LLM to evolve a function to make it more logically complicated.
        
        Args:
            function_code: String containing the current function code
            model: Model to use for evolution
            reasoning_effort: Reasoning effort level for OpenAI reasoning models
            
        Returns:
            Dictionary containing evolved function, stats keys, and evolution description
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                conv = [
                    {"role": "user", "content": prompt_evolve_function.format(
                        original_function=function_code
                    )},
                ]
                response = openai_inference([conv], model=model, return_json=True, reasoning_effort=reasoning_effort)[0]
                
                if response == '[ERROR]':
                    retry_count += 1
                    continue
                    
                json_obj = json.loads(response)
                
                # Validate required fields
                required_fields = ["evolved_function", "stats_keys", "evolution_description"]
                if all(field in json_obj for field in required_fields):
                    evolved_function = json_obj["evolved_function"]
                    stats_keys = json_obj["stats_keys"]
                    evolution_desc = json_obj["evolution_description"]
                    
                    # Validate evolved function is not empty and has proper stats keys
                    if (evolved_function.strip() and 
                        isinstance(stats_keys, list) and 
                        1 <= len(stats_keys) <= 3 and
                        evolution_desc.strip()):
                        return json_obj
                    else:
                        retry_count += 1
                        print(f"    Invalid evolution result format. Retrying... ({retry_count}/{max_retries})")
                else:
                    retry_count += 1
                    missing_fields = [f for f in required_fields if f not in json_obj]
                    print(f"    Missing fields in evolution response: {missing_fields}. Retrying... ({retry_count}/{max_retries})")
                    
            except json.JSONDecodeError as e:
                retry_count += 1
                print(f"    JSON decode error in function evolution (attempt {retry_count}): {e}")
            except Exception as e:
                retry_count += 1
                print(f"    Error in function evolution (attempt {retry_count}): {e}")
        
        # Fallback if all retries failed
        print(f"    Warning: Function evolution failed after {max_retries} attempts")
        return {}
    
    def _execute_function_on_test_cases(self, func_str: str, test_cases: List[Dict], timeout: int) -> Tuple[List[Dict], bool, bool]:
        """Execute a function on test cases and return outputs."""
        # Execute the function
        func_locals = {}
        try:
            exec(func_str, globals(), func_locals)
        except Exception as e:
            return [], True, False
        
        if 'f' not in func_locals:
            return [], True, False
            
        func = func_locals['f']
        
        outputs = []
        has_error = False
        has_timeout = False
        
        for i, case in enumerate(test_cases):
            try:
                inp = case["input"]
                output, stats = self._execute_function_with_timeout(func, inp, timeout)
                
                outputs.append({
                    "case_id": i,
                    "output": output,
                    "stats": stats,
                    "input": case["input"],
                    "success": True
                })
            except TimeoutException:
                has_timeout = True
                outputs.append({
                    "case_id": i,
                    "output": None,
                    "stats": {},
                    "input": case["input"],
                    "success": False,
                    "error": "Timeout"
                })
                break  # Stop on timeout
            except Exception as e:
                has_error = True
                outputs.append({
                    "case_id": i,
                    "output": None,
                    "stats": {},
                    "input": case["input"],
                    "success": False,
                    "error": str(e)
                })
                # Continue processing remaining test cases instead of stopping
        
        return outputs, has_error, has_timeout
    
    def _execute_function_with_timeout(self, func, inp, timeout: int = 5) -> Tuple[Any, Dict]:
        """Execute a function with timeout using multiprocessing."""
        def target(queue, func, inp):
            """Target function for multiprocessing."""
            try:
                # Support dict input (unpack as kwargs), list/tuple (as args), or single value
                if isinstance(inp, dict):
                    result = func(*inp.values())
                elif isinstance(inp, (list, tuple)):
                    result = func(*inp)
                else:
                    result = func(inp)
                queue.put(('success', result))
            except Exception as e:
                queue.put(('error', str(e)))
        
        # Create a queue for communication
        queue = multiprocessing.Queue()
        
        # Create and start the process
        process = multiprocessing.Process(target=target, args=(queue, func, inp))
        process.start()
        
        # Wait for the process to complete or timeout
        process.join(timeout)
        
        if process.is_alive():
            # Process is still running, terminate it
            process.terminate()
            process.join()  # Wait for termination
            raise TimeoutException(f"Function execution timed out (>{timeout}s)")
        
        # Check if we got a result
        if queue.empty():
            raise Exception("Function execution failed without error message")
        
        status, result = queue.get()
        if status == 'success':
            # Extract output and stats if function returns tuple
            if isinstance(result, tuple) and len(result) == 2:
                output, stats = result
                return output, stats
            else:
                return result, {}
        else:
            raise Exception(result) 