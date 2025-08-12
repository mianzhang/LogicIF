"""
LogicIF Framework

A comprehensive framework for generating algorithmic instructions from code functions 
and evaluating language model performance on logical reasoning tasks.
"""

__version__ = "0.1.0"
__author__ = "LogicIF Team"

# Import main classes
from .core import LogicIFGen
from .evaluation import LogicIFEvaluator

# Import main functions
from .inference import run_vllm_inference, main as inference
from .evaluation import main as evaluation

# Create convenience functions
def run_inference(*args, **kwargs):
    """Run VLLM inference. Alias for run_vllm_inference."""
    return run_vllm_inference(*args, **kwargs)

def run_evaluation(*args, **kwargs):
    """Run evaluation with command line interface."""
    import sys
    original_argv = sys.argv
    try:
        # Convert kwargs to command line arguments
        sys.argv = ['evaluation']
        if args:
            sys.argv.extend(args)
        for key, value in kwargs.items():
            if isinstance(value, bool) and value:
                sys.argv.append(f'--{key.replace("_", "-")}')
            elif not isinstance(value, bool):
                sys.argv.extend([f'--{key.replace("_", "-")}', str(value)])
        return evaluation()
    finally:
        sys.argv = original_argv

__all__ = [
    "LogicIFGen",
    "LogicIFEvaluator", 
    "run_vllm_inference",
    "run_inference",
    "run_evaluation",
    "inference",
    "evaluation",
] 