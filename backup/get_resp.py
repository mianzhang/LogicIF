# Copyright 2024 Bytedance Ltd. and/or its affiliates

from dataclasses import dataclass
import os

import llminfer

@dataclass
class QWEN3_32B:
    nickname = "qwen332b"
    model_path = "/localdisk/models/Qwen3-32B"

@dataclass
class CWM:
    nickname = "cwm"
    model_path = "/localdisk/models/facebook/cwm"

@dataclass
class GPT_5:
    nickname = "gpt5"
    model_path = "gpt-5"

@dataclass
class LOGICIFEVALMINI:
    nickname = "logicifevalmini"
    input_key = 'instruction' 
    prompt_path = "benchmark/logic-if-eval-mini.jsonl"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for model in [CWM]:
    for benchmark in [LOGICIFEVALMINI]:
        llminfer.process_jsonl(
            benchmark.prompt_path,
            f'benchmark/{benchmark.nickname}-{model.nickname}.jsonl',
            provider="vllm",
            model=model.model_path,
            input_key=benchmark.input_key,  # Key pointing to string prompts
            max_tokens=16384,
            # max_completion_tokens=16384
        )
