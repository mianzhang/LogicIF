# LogicIF: Complex Logical Instruction Following

LogicIF is a comprehensive framework for generating natural language instructions from code functions and evaluating LLMs performance. The framework consists of four main components:

1. **Instruction Generation Framework** (`LogicIFGen`) (core.py): the framework to generate natural language instructions from code functions.
2. **Benchmarks**:  [LogicIFEval](benchmark/logic-if-eval.jsonl), a collection of 426 instructions with complex logical structures and [LogicIFEval-mini](benchmark/logic-if-eval-mini.jsonl), a compute-friendly version (102 instructions) of  [LogicIFEval](benchmark/logic-if-eval.jsonl).
3. **Inference** (inference.py) & **Evaluation** (evaluation.py): Easy inference and evaluation on generated benchmarks.
#### Installation
```bash
git clone https://github.com/your-org/logicif.git
cd logicif
pip install .
```
#### Prerequisites

Set up OpenAI API key in `config.json` (or use environment variables):
```json
{
    "OPENAI_API_KEY": "your-openai-key",
}
```

## Table of Contents

- [LogicIF: Complex Logical Instruction Following](#logicif-complex-logical-instruction-following)
      - [Installation](#installation)
      - [Prerequisites](#prerequisites)
  - [🚀 Quick Start](#-quick-start)
    - [Evaluate Models on LogicIFEval or LogicIFEval-mini](#evaluate-models-on-logicifeval-or-logicifeval-mini)
      - [Step 1: Get Raw Responses](#step-1-get-raw-responses)
      - [Step 2: Extract Results](#step-2-extract-results)
      - [Get Metrics](#get-metrics)
  - [📝 LogicIFGen: Generating Instructions from Code Functions](#-logicifgen-generating-instructions-from-code-functions)
    - [Main Usage](#main-usage)
      - [1. Prepare Functions and Test Cases](#1-prepare-functions-and-test-cases)
      - [2. Run Instruction Generation](#2-run-instruction-generation)
  - [🔬 Inference](#-inference)
    - [Inference](#inference)
      - [Parameters](#parameters)
  - [📊 Evaluation](#-evaluation)
      - [Extract LLM Results](#extract-llm-results)
      - [Compare Outputs](#compare-outputs)

## 🚀 Quick Start
### Evaluate Models on [LogicIFEval](benchmark/logic-if-eval.jsonl) or [LogicIFEval-mini](benchmark/logic-if-eval-mini.jsonl)
Your model should be supported by `vllm` in order to use our inference script inference.py. Or you need to directly provide a file with the same format as [sample_result.jsonl](sample_result.jsonl) and use the [evaluation.py](evaluation.py) to get the metrics.

For each intruction, we apply a two steps to get the final results (a json object: `{"output": ..., "stats": {...}}`): 
- **Step 1**: get the raw responses of the models (`llm_response` field of [sample_inference_output.jsonl](sample_inference_output.jsonl))
- **Step 2**: use a small OpenAI model to extract the json object. (`llm_output` field of [sample_result.jsonl](sample_result.jsonl))

**Note**: You may skip the Step 1 if you use other inference framework to get the responses and Step 2 if you use other methods to extract the result jsonl object. 

#### Step 1: Get Raw Responses
```
python -m logicif.inference \
    --input "benchmark/logic-if-eval-mini.jsonl" \
    --output "/path/to/inference_output_file" \
    --model "/path/to/your/model" \
    --max-tokens 16384 \
    --gpu-memory-utilization 0.8 \
    --overwrite
```
#### Step 2: Extract Results

Extract structured outputs from raw model responses:

```bash
python -m logicif.evaluation \
    --todo extract_llm_results \
    --input_file "/path/to/inference_output_file" \
    --output_file "/path/to/result_file" \
    --extract_model "gpt-4.1-mini" \ # used in our paper, works well
    --overwrite
```

#### Get Metrics
Compare extracted results against code outputs, including the output accuray, state tracker accuracy, and overall accuracy:

```bash
python -m logicif.evaluation \
    --todo compare_outputs \
    --result_file "/path/to_result_file"
```




## 📝 LogicIFGen: Generating Instructions from Code Functions

LogicIFGen transforms code functions into detailed natural language instructions that can be used for benchmarking language models and model training.

### Main Usage
#### 1. Prepare Functions and Test Cases

1. **Function File:**
   This should be a JSONL file like [sample_functions.jsonl](sample_functions.jsonl), where each line should be a JSON object representing a Python function. The required fields are:
   - `"task_id"`: The ID of the function.
   - `"function"`: The function code as a string.

2. **Test Case File:**
   This should be a JSONL file like [sample_test_cases.jsonl](sample_test_cases.jsonl) where each line contains a JSON object with:
   - `"task_id"`: The function identifier (matching the task_id from function file).
   - `"test_cases"`: A list of test cases, where each test case is a list of arguments to pass to the function.

#### 2. Run Instruction Generation
```python
from logicif import LogicIFGen
from logicif.utils import load_api_keys

# Load API keys before using the framework
load_api_keys()

def main():
    model = "o4-mini" # we recommnend using frontier close-sourced models to guarantee the quality of description
    framework = LogicIFGen(output_dir="functions")
    # Load functions from JSONL file
    framework.load_functions_from_jsonl("sample_functions.jsonl")
    # Add state trackers to functions
    framework.add_stat(model=model, overwrite=True)
    # Evolve functions for more complex logic
    framework.evolve_functions(model=model, max_turns=1, overwrite=True) # you could uncomment this if you do not want change the logic of the functions
    # Generate natural language descriptions
    framework.gen_desc(model=model, overwrite=True)
    # Verify description completeness
    framework.verify_desc(model=model, max_turns=3, overwrite=True)
    # Load test cases
    framework.load_test_cases("sample_test_cases.jsonl")
    # Execute functions to get expected outputs
    framework.get_code_output(timeout=5)
    # Analyze complexity
    framework.analyze_complexity()
    # Generate final benchmark file
    framework.finalize_benchmark(output_file="sample_instructions.jsonl")

if __name__ == "__main__":
    main()
```

**Generated Instruction**

The genereted file is like `sample_instructions.jsonl`. Each line contains a complete instruction example with the following fields:

- `"task_id"`: The function identifier
- `"test_case_id"`: Index of the specific test case (0, 1, 2, ...)
- `"input"`: The input values for this test case
- `"code_output"`: The expected output and the value of state trackers from running the function
- `"function"`: The function code with state trackers
- `"description"`: Natural language description of the code function
- `"instruction"`: The complete instruction prompt
- `"complexity_score"`: Complexity score for the function


## 🔬 Inference

The inference module uses `vllm` to run language model inference on benchmark files. If you want to evaluate the models on custom functions, please follow the previous section to generation instructions. You could also evaluate the models on `LogicIFEval` or `LogicIFEval-mini` located in `benchmark/`, which share the same format as `sample_instructions.jsonl`.

### Inference
```bash
python -m logicif.inference \
    --input "sample_instructions.jsonl" \
    --output "sample_inference_output.jsonl" \
    --model "/path/to/your/model" \
    --max-tokens 16384 \
    --gpu-memory-utilization 0.8 \
    --overwrite
```

#### Parameters

- `--input`: Input benchmark JSONL file with instructions
- `--output`: Output file to save model responses
- `--model`: HuggingFace model ID or local path
- `--max-tokens`: Maximum tokens to generate (default: 16384)
- `--run-name`: Optional run name for experiment tracking
- `--overwrite`: Overwrite existing output file
- `--gpu-memory-utilization`: GPU memory utilization ratio (default: 0.8)

## 📊 Evaluation

The evaluation module provides tools to extract model outputs and compare them against expected results. In our paper, we use `gpt-4.1-mini` as the json extractor.

#### Extract LLM Results

Extract structured outputs from raw model responses:

```bash
python -m logicif.evaluation \
    --todo extract_llm_results \
    --input_file "sample_inference_output.jsonl" \
    --output_file "sample_result.jsonl" \
    --extract_model "gpt-4.1-mini" \
    --overwrite
```
**Parameters**
- `--input_file`: JSONL file with raw model responses
- `--output_file`: Output file for extracted results
- `--extract_model`: Model to use for output extraction
- `--overwrite`: Overwrite existing files

#### Compare Outputs

Compare extracted results against expected outputs:

```bash
python -m logicif.evaluation \
    --todo compare_outputs \
    --result_file "sample_result.jsonl"
```

**Parameters**
- `--result_file`: Path to file with extracted results

