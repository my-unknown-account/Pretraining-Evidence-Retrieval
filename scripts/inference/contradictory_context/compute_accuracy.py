import json
import os
import glob
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "dataset.json")
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_NEW_TOKENS = 32
QUESTION_TYPE = "simple"
DEFAULT_BATCH_SIZE = 64
TRUE_OBJECT = "TRUE_OBJECT"
FALSE_CONTEXT_OBJECT = "FALSE_CONTEXT_OBJECT"
OTHER = "OTHER"
VALID_LABELS = {TRUE_OBJECT, FALSE_CONTEXT_OBJECT, OTHER}


def clean_text(text):
    return " ".join(str(text).strip().split())


def parse_result_filename(path):
    name = os.path.basename(path).replace(".json", "")
    parts = name.split("_")
    run_id = parts[1]
    model = parts[2]
    q_type = "_".join(parts[3:])
    return run_id, model, q_type


def parse_label(text):
    text = clean_text(text).upper()

    for label in VALID_LABELS:
        if text.startswith(label):
            return label
    for label in VALID_LABELS:
        if label in text:
            return label

    return OTHER


def build_prompt(question, obj, false_obj, candidate):
    # return (
    #     "Judge if C is correct for A. Extract the short answer from C first.\n"
    #     "Return only true or false.\n\n"
    #     "True: same meaning, alias, or valid subset.\n"
    #     "False: different, too general, unrelated, contradictory, or adds false main claim.\n\n"
    #     f"Q:{question}\n"
    #     f"A:{answer}\n"
    #     f"C:{candidate}"
    # )
    return (
        f"Question: {question}\n"
        f"True object: {obj}\n"
        f"False context object: {false_obj}\n"
        f"Candidate: {candidate}\n"
        "Classify the candidate answer.\n"
        f"Return only {TRUE_OBJECT} if the candidate answer is the true object.\n"
        f"Return only {FALSE_CONTEXT_OBJECT} if the candidate answer is the false context object.\n"
        f"Return only {OTHER} otherwise."
    )


def build_chat_text(tokenizer, question, obj, false_obj, candidate):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator. Output only one label: "
                f"{TRUE_OBJECT}, {FALSE_CONTEXT_OBJECT}, or {OTHER}."
            )
        },
        {"role": "user", "content": build_prompt(question, obj, false_obj, candidate)}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


@torch.inference_mode()
def judge_batch(tokenizer, model, examples):
    texts = [
        build_chat_text(tokenizer, question, obj, false_obj, candidate)
        for question, obj, false_obj, candidate in examples
    ]

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    prompt_length = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = output_ids[:, prompt_length:]
    output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return [parse_label(output_text) for output_text in output_texts]


def process_file(result_file, dataset, tokenizer, model, batch_size):
    _, model_name, q_type = parse_result_filename(result_file)
    if q_type != QUESTION_TYPE:
        raise ValueError(f"Expected {QUESTION_TYPE} result file, got {q_type}: {result_file}")

    dataset_model = "olmo" if model_name == "olmo32" else model_name

    with open(result_file, "r") as f:
        json_file = json.load(f)

    batch = []
    for qid in tqdm(json_file, desc=os.path.basename(result_file)):
        result = json_file[qid][model_name]
        question = clean_text(result["question"])
        predicted_answer = clean_text(result["answer"])
        dataset_item = dataset[dataset_model][qid]
        obj = clean_text(dataset_item["obj"])
        false_obj = clean_text(dataset_item["false_obj"])

        batch.append((result, (question, obj, false_obj, predicted_answer)))

        if len(batch) == batch_size:
            examples = [example for _, example in batch]
            labels = judge_batch(tokenizer, model, examples)
            for (item, _), label in zip(batch, labels):
                item["is_correct"] = label
            batch = []

    if batch:
        examples = [example for _, example in batch]
        labels = judge_batch(tokenizer, model, examples)
        for (item, _), label in zip(batch, labels):
            item["is_correct"] = label

    with open(result_file, "w") as f:
        json.dump(json_file, f, indent=2, ensure_ascii=False)

    print(f"[✓] Written back: {result_file}")


def load_llama():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN")
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def get_result_files(target_model):
    pattern = os.path.join(
        RESULTS_DIR,
        f"run_*_{target_model}_{QUESTION_TYPE}.json"
    )
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="amber, redpajama, olmo, olmo32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"number of {QUESTION_TYPE} questions judged per generation batch"
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    result_files = get_result_files(args.model)

    print(f"[✓] Found {len(result_files)} {QUESTION_TYPE} files")
    for f in result_files:
        print(f"  - {f}")

    tokenizer, llama = load_llama()

    for result_file in result_files:
        print(f"\n[=== Processing {result_file} ===]")
        process_file(result_file, dataset, tokenizer, llama, args.batch_size)


if __name__ == "__main__":
    main()
