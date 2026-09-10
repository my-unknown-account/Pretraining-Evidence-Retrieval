import json
import os
import glob
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset.json")
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_NEW_TOKENS = 32


def clean_text(text):
    return " ".join(str(text).strip().split())


def parse_result_filename(path):
    name = os.path.basename(path).replace(".json", "")
    parts = name.split("_")
    run_id = parts[1]
    model = parts[2]
    q_type = "_".join(parts[3:])
    return run_id, model, q_type


def parse_bool(text):
    text = text.strip().lower()

    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False
    if "true" in text and "false" not in text:
        return True
    if "false" in text and "true" not in text:
        return False

    return False


def build_prompt(question, answer, candidate):
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
        f"Answer: {answer}\n"
        f"Candidate: {candidate}\n"
        "Is candidate correct?"
    )


@torch.inference_mode()
def judge(tokenizer, model, question, answer, candidate):
    messages = [
        {"role": "system", "content": "You are a strict evaluator. Output only true or false."},
        {"role": "user", "content": build_prompt(question, answer, candidate)}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return parse_bool(output_text)


def process_file(result_file, dataset, tokenizer, model):
    run_id, model_name, q_type = parse_result_filename(result_file)
    dataset_model = "olmo" if model_name == "olmo32" else model_name

    with open(result_file, "r") as f:
        json_file = json.load(f)

    for qid in tqdm(json_file, desc=os.path.basename(result_file)):
        question = clean_text(json_file[qid][model_name][q_type]["question"])
        predicted_answer = clean_text(json_file[qid][model_name][q_type]["answer"])
        ground_truth = clean_text(dataset[dataset_model][qid]["obj"])

        json_file[qid][model_name][q_type]["is_correct"] = judge(
            tokenizer,
            model,
            question,
            ground_truth,
            predicted_answer
        )

    with open(result_file, "w") as f:
        json.dump(json_file, f, indent=2, ensure_ascii=False)

    print(f"[✓] Written back: {result_file}")


def load_llama():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN")
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def get_result_files(target_model, target_q_type):
    pattern = os.path.join(
        RESULTS_DIR,
        f"run_*_{target_model}_{target_q_type}.json"
    )
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="amber, redpajama, olmo, olmo32")
    parser.add_argument("--q_type", required=True, help="simple, complex, template_based")
    args = parser.parse_args()

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    result_files = get_result_files(args.model, args.q_type)

    print(f"[✓] Found {len(result_files)} files")
    for f in result_files:
        print(f"  - {f}")

    tokenizer, llama = load_llama()

    for result_file in result_files:
        print(f"\n[=== Processing {result_file} ===]")
        process_file(result_file, dataset, tokenizer, llama)


if __name__ == "__main__":
    main()