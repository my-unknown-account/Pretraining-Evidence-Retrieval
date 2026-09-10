import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)


# ==== STATIC CONFIG ====
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]
DATASET_PATH = ROOT_DIR / "data" / "dataset.json"
OUTPUT_DIR = BASE_DIR / "results"

MAX_NEW_TOKENS = 128
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
# ======================


MODEL_MAP = {
    "amber": "LLM360/AmberChat",
    "redpajama": "togethercomputer/RedPajama-INCITE-7B-Instruct",
    "olmo": "allenai/Olmo-3-7B-Instruct",
    "olmo32": "allenai/Olmo-3.1-32B-Instruct",
}


AMBER_TEMPLATE = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
### Human: Got any creative ideas for a 10 year old’s birthday?
### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments.
3. Outdoor Movie Night: Set up a backyard movie night with a projector.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity.
5. Sports Olympics: Host a mini Olympics event.
6. Cooking Party: Have a cooking-themed party.
7. Superhero Training Camp: Create a superhero-themed party.
8. Outdoor Adventure: Plan an outdoor adventure party.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!
### Human: {prompt}
### Assistant:"""


def clean_answer(text: str) -> str:
    text = text.strip()

    stop_markers = [
        "### Human:",
        "### Assistant:",
        "\nQuestion:",
        "\nAnswer:",
        "Question:",
        "Answer:",
        "</s>",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    text = text.split("\n")[0].strip()
    text = text.strip(" .,:;\"'")

    return text


def load_model(model_key: str):
    model_name = MODEL_MAP[model_key]

    if model_key == "amber":
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    model.eval()

    return tokenizer, model


def format_prompt(model_key: str, tokenizer, question: str) -> str:
    if model_key == "amber":
        return AMBER_TEMPLATE.format(prompt=question)

    if model_key == "redpajama":
        return f"Q: {question}\nA:"

    if model_key in {"olmo", "olmo32"}:
        messages = [
            {
                "role": "user",
                "content": (
                    f'Do not say "As of my last update" and something like it. '
                    f"Just Answer. {question}\n"
                ),
            }
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    raise ValueError(f"Unknown model: {model_key}")


def format_verbalized_confidence_prompt(question: str, answer: str) -> str:
    return f"""Question: {question}
Answer: {answer}

How confident are you that the answer is correct?
Respond only with a number from 0 to 100.

Confidence:"""


def format_ptrue_prompt(question: str, answer: str) -> str:
    return f"""Question: {question}
Answer: {answer}

Is the answer correct?
Reply only True or False.

True or False:"""


@torch.no_grad()
def generate_batch(
    tokenizer,
    model,
    prompts,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
):
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_len = inputs.input_ids.shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    generated_ids = outputs[:, input_len:]

    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return [clean_answer(x) for x in decoded]


@torch.no_grad()
def mean_token_logprob(tokenizer, model, prompt: str, answer: str):
    device = next(model.parameters()).device

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    full_ids = tokenizer(
        prompt + " " + answer,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]

    logprobs = F.log_softmax(logits, dim=-1)

    mask = target != -100
    token_logprobs = logprobs.gather(
        dim=-1,
        index=target.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)

    selected = token_logprobs[mask]

    if selected.numel() == 0:
        return {
            "sum_logprob": None,
            "mean_logprob": None,
            "geometric_mean_prob": None,
            "num_tokens": 0,
        }

    sum_lp = selected.sum().item()
    mean_lp = selected.mean().item()

    return {
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "geometric_mean_prob": float(np.exp(mean_lp)),
        "num_tokens": int(selected.numel()),
    }


@torch.no_grad()
def continuation_logprob(tokenizer, model, prompt: str, continuation: str):
    device = next(model.parameters()).device

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    full_ids = tokenizer(
        prompt + continuation,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]

    logprobs = F.log_softmax(logits, dim=-1)

    mask = target != -100
    token_logprobs = logprobs.gather(
        dim=-1,
        index=target.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)

    selected = token_logprobs[mask]

    if selected.numel() == 0:
        return None

    return selected.sum().item()


def parse_confidence(text: str):
    match = re.search(r"(\d+(?:\.\d+)?)", text)

    if match is None:
        return None

    value = float(match.group(1))

    if value > 1.0:
        value = value / 100.0

    return max(0.0, min(1.0, value))


@torch.no_grad()
def verbalized_confidence(tokenizer, model, question: str, answer: str):
    prompt = format_verbalized_confidence_prompt(question, answer)

    output = generate_batch(
        tokenizer=tokenizer,
        model=model,
        prompts=[prompt],
        max_new_tokens=8,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )[0]

    return {
        "prompt": prompt,
        "raw_output": output,
        "confidence_0_to_1": parse_confidence(output),
    }


@torch.no_grad()
def p_true_score(tokenizer, model, question: str, answer: str):
    prompt = format_ptrue_prompt(question, answer)

    true_lp = continuation_logprob(tokenizer, model, prompt, " True")
    false_lp = continuation_logprob(tokenizer, model, prompt, " False")

    if true_lp is None or false_lp is None:
        return {
            "prompt": prompt,
            "p_true": None,
            "p_false": None,
            "true_logprob": true_lp,
            "false_logprob": false_lp,
        }

    probs = torch.softmax(
        torch.tensor([true_lp, false_lp], dtype=torch.float32),
        dim=0,
    )

    return {
        "prompt": prompt,
        "p_true": probs[0].item(),
        "p_false": probs[1].item(),
        "true_logprob": true_lp,
        "false_logprob": false_lp,
    }


def load_existing_results(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_result(
    results,
    qid,
    model_key,
    question,
    generation_prompt,
    answer,
    token_logprob,
    verbalized_conf,
    p_true,
):
    if qid not in results:
        results[qid] = {}

    if model_key not in results[qid]:
        results[qid][model_key] = {}

    results[qid][model_key] = {
        "question": question,
        "generation_prompt": generation_prompt,
        "answer": answer,
        "confidence": {
            "token_logprob": token_logprob,
            "verbalized_confidence": verbalized_conf,
            "p_true": p_true,
        },
    }


def run(args):
    model_key = args.model
    dataset_key = args.model

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    dataset_key = 'olmo' if dataset_key == 'olmo32' else dataset_key
    if dataset_key not in all_data:
        raise ValueError(
            f"Dataset key '{dataset_key}' not found. "
            f"Available keys: {list(all_data.keys())}"
        )

    data = all_data[dataset_key]

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"run_{args.run_id}_{model_key}_simple.json"
    )

    results = load_existing_results(output_path)

    tokenizer, model = load_model(model_key)

    qids = list(data.keys())

    for i in tqdm(range(0, len(qids), args.batch_size)):
        batch_qids = qids[i : i + args.batch_size]

        batch_info = []
        batch_prompts = []

        for qid in batch_qids:
            item = data[qid]

            if "question" not in item:
                raise ValueError(
                    f"question not found for {qid}. "
                    f"Available: {list(item.keys())}"
                )

            question = item["question"]
            prompt = format_prompt(model_key, tokenizer, question)

            batch_info.append((qid, question))
            batch_prompts.append(prompt)

        answers = generate_batch(
            tokenizer=tokenizer,
            model=model,
            prompts=batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        for (qid, question), generation_prompt, answer in zip(
            batch_info,
            batch_prompts,
            answers,
        ):
            token_lp = mean_token_logprob(
                tokenizer=tokenizer,
                model=model,
                prompt=generation_prompt,
                answer=answer,
            )

            verbal_conf = verbalized_confidence(
                tokenizer=tokenizer,
                model=model,
                question=question,
                answer=answer,
            )

            p_true = p_true_score(
                tokenizer=tokenizer,
                model=model,
                question=question,
                answer=answer,
            )

            save_result(
                results=results,
                qid=qid,
                model_key=model_key,
                question=question,
                generation_prompt=generation_prompt,
                answer=answer,
                token_logprob=token_lp,
                verbalized_conf=verbal_conf,
                p_true=p_true,
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
    )

    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
