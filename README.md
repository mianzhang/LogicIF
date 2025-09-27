# LogicIF: Complex Logical Instruction Following

## Updates
- **Sep 25.** We replaced the JSON object extraction via OpenAI with a rule-based JSON extractor, which makes the evaluation process mush easier!
- **Aug 18.** LogicIFEval is now available on Hugging Face Datasets! You can easily explore task characteristics using the built-in dataset viewer: 👉 [https://huggingface.co/datasets/mianzhang/LogicIFEval](https://huggingface.co/datasets/billmianz/LogicIFEval)

---

## Evaluate Models on [LogicIFEval](benchmark/logic-if-eval.jsonl) or [LogicIFEval-mini](benchmark/logic-if-eval-mini.jsonl)
The instructions are provided in the `instruction` field of the benchmark files. You need to create the a output file with at least the following fields:
- `"task_id"`: The ID of the function.
- `"test_case_id"`: The ID of the test case.
- `"code_output"`: The output from code function.
- `"response"`: Models' response to the `instruction`.

The output file with the responses of gpt-5 and [facebook/cwm](https://huggingface.co/facebook/cwm) can be found in `benchmark/`.

Then, call [evaluation.py](evaluation.py) to get the metrics:
```bash
python evaluation --result_file benchmark/logicifevalmini-cwm.jsonl
# Case-level Accuracy: 0.8144192256341789
# Question-level Accuracy: 0.47058823529411764
```

---

## LogicIFGen: Generating Instructions from Code Functions

LogicIFGen transforms code functions into detailed natural language instructions that can be used for benchmarking language models and model training.

### Installation
```bash
git clone https://github.com/mianzhang/LogicIF
cd logicif
pip install -e .
```

### Prerequisites

Set up OpenAI API key in `config.json` (or use environment variables):
```json
{
    "OPENAI_API_KEY": "your-openai-key"
}
```

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


## Contact

Please email Mian (`mian.zhang@utdallas.edu`) if you have any questions. If you encounter any issues with the code, or data, please open an issue on GitHub.

## Citation

```bibtex
@article{zhang2025complex,
  title={Complex Logical Instruction Generation},
  author={Zhang, Mian and Liu, Shujian and Dong, Sixun and Yin, Ming and Hu, Yebowen and Wang, Xun and Ma, Steven and Wang, Song and Indurthi, Sathish Reddy and Deng, Haoyun and others},
  journal={arXiv preprint arXiv:2508.09125},
  year={2025}
}
```
