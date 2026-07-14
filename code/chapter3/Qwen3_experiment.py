"""Qwen3-0.6B 金融信息抽取实验。

示例：
    python Qwen3_experiment.py --mode smoke
    python Qwen3_experiment.py --mode all
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".cache" / "huggingface"))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_RESULTS_PATH = SCRIPT_DIR / "qwen3_experiment_results.json"
REPEATS = 3
MAX_NEW_TOKENS = 256

ANNOUNCEMENT = (
    "【虚构测试公告】星海科技控股有限公司（09999.HK）公告：公司于2026年7月10日"
    "在香港交易所回购1,200,000股普通股，每股价格介于450.20港元至456.80港元，"
    "总代价约544,000,000港元。公告未披露股份注销日期。"
)

GOLD_ANSWER: dict[str, Any] = {
    "company": "星海科技控股有限公司",
    "stock_code": "09999.HK",
    "event_type": "股份回购",
    "transaction_date": "2026-07-10",
    "share_count": 1_200_000,
    "price_min": 450.20,
    "price_max": 456.80,
    "total_amount": 544_000_000,
    "currency": "HKD",
    "cancellation_date": None,
    "undisclosed_fields": ["cancellation_date"],
}

REQUIRED_FIELDS = list(GOLD_ANSWER)

SAMPLING_CONFIGS = {
    "greedy": {"do_sample": False},
    "balanced": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    },
    "creative": {
        "do_sample": True,
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
    },
}

SYSTEM_PROMPT = """你是金融公告信息抽取助手。只能使用公告中明确提供的事实，不得猜测。
未披露的字段必须填 null，并在 undisclosed_fields 数组中列出字段名。
最终只输出一个合法 JSON 对象，不要输出 Markdown、解释或分析过程。"""

SCHEMA_INSTRUCTION = """请严格输出以下字段：
company, stock_code, event_type, transaction_date, share_count, price_min,
price_max, total_amount, currency, cancellation_date, undisclosed_fields。
日期格式为 YYYY-MM-DD；金额和股数使用数字；币种使用 ISO 代码。"""


def build_user_prompt(strategy: str) -> str:
    """为三种提示策略生成输入。"""
    if strategy == "zero_shot":
        return f"{SCHEMA_INSTRUCTION}\n\n请从下面的公告中抽取信息：\n{ANNOUNCEMENT}"

    if strategy == "few_shot":
        example_text = (
            "【虚构示例】远洋零售有限公司（08888.HK）于2026年6月1日回购10,000股，"
            "每股12.50港元，总代价125,000港元，公告未披露股份注销日期。"
        )
        example_answer = {
            "company": "远洋零售有限公司",
            "stock_code": "08888.HK",
            "event_type": "股份回购",
            "transaction_date": "2026-06-01",
            "share_count": 10_000,
            "price_min": 12.50,
            "price_max": 12.50,
            "total_amount": 125_000,
            "currency": "HKD",
            "cancellation_date": None,
            "undisclosed_fields": ["cancellation_date"],
        }
        return (
            f"{SCHEMA_INSTRUCTION}\n\n示例公告：\n{example_text}\n示例答案：\n"
            f"{json.dumps(example_answer, ensure_ascii=False)}\n\n"
            f"现在请抽取目标公告：\n{ANNOUNCEMENT}"
        )

    if strategy == "stepwise_cot":
        return (
            f"{SCHEMA_INSTRUCTION}\n\n请在内部按字段逐一核对原文证据，检查缺失值、数字、"
            "日期和币种。不要展示推理过程，最终只输出 JSON。\n\n"
            f"目标公告：\n{ANNOUNCEMENT}"
        )

    raise ValueError(f"未知 Prompt 策略：{strategy}")


def extract_json(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    """从模型最终回答中提取一个 JSON 对象。"""
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, "未找到 JSON 对象"
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    if not isinstance(parsed, dict):
        return None, "JSON 顶层不是对象"
    return parsed, None


def values_equal(actual: Any, expected: Any) -> bool:
    """允许数字 int/float 表示差异，其他值严格比较。"""
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-6
    return actual == expected


def evaluate(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """将模型 JSON 与标准答案自动比对。"""
    if parsed is None:
        return {
            "json_valid": False,
            "correct_fields": 0,
            "field_accuracy": 0.0,
            "present_required_fields": 0,
            "required_completeness": 0.0,
            "hallucination_fields": 0,
            "incorrect_fields": REQUIRED_FIELDS,
        }

    correct = [
        field
        for field, expected in GOLD_ANSWER.items()
        if field in parsed and values_equal(parsed[field], expected)
    ]
    present = [field for field in REQUIRED_FIELDS if field in parsed]
    incorrect = [field for field in REQUIRED_FIELDS if field not in correct]
    unexpected = set(parsed) - set(GOLD_ANSWER)
    contradicted_unknown = int(parsed.get("cancellation_date") is not None)
    hallucination_fields = len(unexpected) + contradicted_unknown
    return {
        "json_valid": True,
        "correct_fields": len(correct),
        "field_accuracy": len(correct) / len(GOLD_ANSWER),
        "present_required_fields": len(present),
        "required_completeness": len(present) / len(REQUIRED_FIELDS),
        "hallucination_fields": hallucination_fields,
        "incorrect_fields": incorrect,
        "unexpected_fields": sorted(unexpected),
    }


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model() -> tuple[Any, Any, str, float]:
    """加载模型，MPS 初始化失败时回退 CPU。"""
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    device = choose_device()
    if device == "mps":
        try:
            model = model.to(device)
        except (RuntimeError, NotImplementedError) as exc:
            print(f"MPS 初始化失败，回退 CPU：{exc}", file=sys.stderr)
            device = "cpu"
            model = model.to(device)
    model.eval()
    return tokenizer, model, device, time.perf_counter() - started


def generate_once(
    tokenizer: Any,
    model: Any,
    device: str,
    experiment_type: str,
    prompt_strategy: str,
    sampling_name: str,
    repeat: int,
) -> tuple[dict[str, Any], str]:
    """执行一次生成；若 MPS 运行失败则移动模型到 CPU 后重试。"""
    user_prompt = build_user_prompt(prompt_strategy)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[-1])
    seed = 41 + repeat
    torch.manual_seed(seed)

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.eos_token_id,
        **SAMPLING_CONFIGS[sampling_name],
    }

    def run_on(target_device: str) -> tuple[Any, float]:
        target_inputs = {key: value.to(target_device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**target_inputs, **generation_kwargs)
        elapsed = time.perf_counter() - started
        generated_ids = output_ids[0, input_tokens:].detach().cpu()
        return generated_ids, elapsed

    run_device = device
    try:
        generated_ids, elapsed = run_on(run_device)
    except (RuntimeError, NotImplementedError) as exc:
        if run_device != "mps":
            raise
        print(f"MPS 生成失败，回退 CPU：{exc}", file=sys.stderr)
        model.to("cpu")
        run_device = "cpu"
        generated_ids, elapsed = run_on(run_device)

    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed, parse_error = extract_json(raw_output)
    metrics = evaluate(parsed)
    result = {
        "experiment_type": experiment_type,
        "prompt_strategy": prompt_strategy,
        "sampling_strategy": sampling_name,
        "repeat": repeat + 1,
        "seed": seed,
        "device": run_device,
        "parameters": generation_kwargs,
        "input_tokens": input_tokens,
        "output_tokens": int(generated_ids.shape[-1]),
        "latency_seconds": elapsed,
        "raw_output": raw_output,
        "parsed_output": parsed,
        "parse_error": parse_error,
        "metrics": metrics,
    }
    return result, run_device


def normalized_output(run: dict[str, Any]) -> str:
    parsed = run["parsed_output"]
    if parsed is None:
        return run["raw_output"].strip()
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (
            run["experiment_type"],
            run["prompt_strategy"],
            run["sampling_strategy"],
        )
        groups.setdefault(key, []).append(run)

    summaries = []
    for (experiment_type, prompt_strategy, sampling_strategy), items in groups.items():
        count = len(items)
        outputs = Counter(normalized_output(item) for item in items)
        summaries.append(
            {
                "experiment_type": experiment_type,
                "prompt_strategy": prompt_strategy,
                "sampling_strategy": sampling_strategy,
                "runs": count,
                "json_parse_success_rate": sum(
                    item["metrics"]["json_valid"] for item in items
                )
                / count,
                "average_field_accuracy": sum(
                    item["metrics"]["field_accuracy"] for item in items
                )
                / count,
                "average_required_completeness": sum(
                    item["metrics"]["required_completeness"] for item in items
                )
                / count,
                "total_hallucination_fields": sum(
                    item["metrics"]["hallucination_fields"] for item in items
                ),
                "output_consistency": max(outputs.values()) / count,
                "average_input_tokens": sum(item["input_tokens"] for item in items)
                / count,
                "average_output_tokens": sum(item["output_tokens"] for item in items)
                / count,
                "average_latency_seconds": sum(
                    item["latency_seconds"] for item in items
                )
                / count,
            }
        )
    return summaries


def experiment_matrix(mode: str) -> list[tuple[str, str, str, int]]:
    if mode == "smoke":
        return [("smoke", "zero_shot", "balanced", 0)]

    matrix: list[tuple[str, str, str, int]] = []
    if mode in {"sampling", "all"}:
        for sampling in SAMPLING_CONFIGS:
            for repeat in range(REPEATS):
                matrix.append(("sampling", "zero_shot", sampling, repeat))
    if mode in {"prompt", "all"}:
        for prompt in ("zero_shot", "few_shot", "stepwise_cot"):
            for repeat in range(REPEATS):
                matrix.append(("prompt", prompt, "balanced", repeat))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "sampling", "prompt", "all"),
        default="smoke",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    tokenizer, model, device, load_seconds = load_model()
    runs = []
    matrix = experiment_matrix(args.mode)
    for index, (
        experiment_type,
        prompt_strategy,
        sampling_strategy,
        repeat,
    ) in enumerate(matrix, 1):
        print(
            f"[{index}/{len(matrix)}] prompt={prompt_strategy}, "
            f"sampling={sampling_strategy}, repeat={repeat + 1}"
        )
        result, device = generate_once(
            tokenizer,
            model,
            device,
            experiment_type,
            prompt_strategy,
            sampling_strategy,
            repeat,
        )
        runs.append(result)

    payload = {
        "experiment": "Qwen3-0.6B synthetic HK stock buyback extraction",
        "notice": "所有公告、公司、代码及数据均为虚构，仅用于课程实验。",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": MODEL_ID,
        "model_config_version": getattr(model.config, "transformers_version", None),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "initial_device": choose_device(),
            "final_device": device,
            "model_load_seconds": load_seconds,
            "hf_home": os.environ["HF_HOME"],
        },
        "settings": {
            "enable_thinking": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repeats": REPEATS,
        },
        "input": ANNOUNCEMENT,
        "gold_answer": GOLD_ANSWER,
        "runs": runs,
        "summary": summarize(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"结果已保存：{args.output}")


if __name__ == "__main__":
    main()
