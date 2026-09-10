from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Relation configuration
# ---------------------------------------------------------------------------

RELATION_TEMPLATES: Dict[str, str] = {
    "P106": "The occupation of {s} is {o}.",
    "P19": "{s} was born in {o}.",
    "P136": "The genre of {s} is {o}.",
    "P22": "The father of {s} is {o}.",
    "P17": "{s} is located in {o}.",
    "P162": "{s} was produced by {o}.",
    "P57": "{s} was directed by {o}.",
    "P1376": "{s} is the capital of {o}.",
    "P58": "{s} was written by {o}.",
    "P86": "{s} was composed by {o}.",
    "P140": "The religion of {s} is {o}.",
    "P641": "The sport of {s} is {o}.",
    "P50": "{s} was authored by {o}.",
    "P25": "The mother of {s} is {o}.",
    "P36": "The capital of {s} is {o}.",
}

RELATION_NAMES: Dict[str, str] = {
    "P106": "occupation",
    "P19": "place of birth",
    "P136": "genre",
    "P22": "father",
    "P17": "country",
    "P162": "producer",
    "P57": "director",
    "P1376": "capital of",
    "P58": "screenwriter",
    "P86": "composer",
    "P140": "religion",
    "P641": "sport",
    "P50": "author",
    "P25": "mother",
    "P36": "capital",
}

HIGH_RISK_MULTI_VALUED: Set[str] = {
    "P106", "P136", "P162", "P57", "P58",
    "P86", "P140", "P641", "P50",
}

RELATION_NOTES: Dict[str, str] = {
    "P106": "Occupation is often multi-valued. Search carefully for evidence that the candidate occupation may also be true.",
    "P19": "Check birthplace aliases, historical administrative names, and disputed birthplace claims.",
    "P136": "Genre is often multi-valued/hierarchical. Another genre, parent genre, or subgenre may also be valid.",
    "P22": "Check disputed genealogy and biological/adoptive ambiguity.",
    "P17": "Check historical-state, sovereignty, and territorial ambiguity.",
    "P162": "Works often have multiple producers.",
    "P57": "Works may have co-directors.",
    "P1376": "Check historical and disputed capital status.",
    "P58": "Works often have multiple screenwriters.",
    "P86": "Works can have multiple composers.",
    "P140": "Religion can be multi-valued, changed over time, or disputed.",
    "P641": "A person/event may be associated with multiple sports.",
    "P50": "Works may have multiple authors.",
    "P25": "Check disputed genealogy and biological/adoptive ambiguity.",
    "P36": "A polity can have multiple, official, de facto, legislative, or historical capitals.",
}


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

GENERATION_FORMAT = {
    "type": "json_schema",
    "name": "ranked_contradiction_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["CANDIDATES", "UNSUITABLE"],
            },
            "candidates": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "qid": {"type": "string"},
                        "label": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["qid", "label", "reason"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["status", "candidates", "reason"],
        "additionalProperties": False,
    },
}

VERIFY_FORMAT = {
    "type": "json_schema",
    "name": "positive_evidence_check",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": [
                    "PASS_NO_POSITIVE_EVIDENCE",
                    "REJECT_TRUE_OR_ALTERNATIVE",
                    "REJECT_HISTORICAL_OR_DISPUTED",
                    "REJECT_TYPE_OR_GRANULARITY",
                ],
            },
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    data = dict()
    with open(path, 'r') as f:
        dataset = json.load(f)
    for _model in dataset:
        for qid in dataset[_model]:
            if qid not in data:
                data[qid] = dataset[_model][qid]
    return data


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalize_label(x: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", x.casefold())).strip()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def chunks(xs: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


# ---------------------------------------------------------------------------
# Dataset indexes
# ---------------------------------------------------------------------------

def build_indexes(
    data: Mapping[str, Mapping[str, Any]],
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[Tuple[str, str], Set[str]],
]:
    relation_pool: Dict[str, Dict[str, str]] = defaultdict(dict)
    dataset_truth: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for item in data.values():
        pid = str(item.get("rel_entity", "")).strip()
        sq = str(item.get("sub_entity", "")).strip()
        oq = str(item.get("obj_entity", "")).strip()
        ol = str(item.get("obj", "")).strip()

        if pid not in RELATION_TEMPLATES:
            continue
        if not (sq.startswith("Q") and oq.startswith("Q")):
            continue

        relation_pool[pid].setdefault(oq, ol)
        dataset_truth[(sq, pid)].add(oq)

    return dict(relation_pool), dict(dataset_truth)


# ---------------------------------------------------------------------------
# Wikidata claims
# ---------------------------------------------------------------------------

def extract_entity_qids(claims: Mapping[str, Any], pid: str) -> Set[str]:
    out: Set[str] = set()
    for claim in claims.get(pid, []) or []:
        snak = claim.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        dv = snak.get("datavalue") or {}
        val = dv.get("value")
        if isinstance(val, dict):
            qid = val.get("id")
            if isinstance(qid, str) and qid.startswith("Q"):
                out.add(qid)
    return out


def preload_wikidata_truth(
    data: Mapping[str, Mapping[str, Any]],
    cache_path: Path,
    batch_size: int = 50,
    max_retries: int = 5,
) -> Dict[str, Dict[str, List[str]]]:
    cache = read_json(cache_path) if cache_path.exists() else {}

    subjects: Set[str] = set()
    pids: Set[str] = set()

    for item in data.values():
        sq = str(item.get("sub_entity", "")).strip()
        pid = str(item.get("rel_entity", "")).strip()
        if sq.startswith("Q") and pid in RELATION_TEMPLATES:
            subjects.add(sq)
            pids.add(pid)

    missing = sorted(q for q in subjects if q not in cache)
    if not missing:
        return cache

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ECIR-ContradictionBuilder/2.0 (academic research)"
    })

    def retry_delay(response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), 120.0)
                except ValueError:
                    pass
        return min(2 ** attempt, 60) + random.uniform(0, 1.5)

    def fetch_batch(batch: Sequence[str]) -> Dict[str, Any] | None:
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "claims",
            "format": "json",
        }
        last_err = None
        for k in range(max_retries):
            response = None
            try:
                response = session.get(
                    "https://www.wikidata.org/w/api.php",
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_err = e
                time.sleep(retry_delay(response, k))
        print(f"[WARNING] Wikidata batch failed after retries: {last_err}")
        return None

    def cache_entities(batch: Sequence[str], payload: Mapping[str, Any]) -> None:
        entities = payload.get("entities", {})
        for qid in batch:
            entity = entities.get(qid, {})
            claims = entity.get("claims", {}) if isinstance(entity, dict) else {}
            cache[qid] = {
                pid: sorted(extract_entity_qids(claims, pid))
                for pid in pids
                if pid in claims
            }

    done = 0
    failed_batches: List[Sequence[str]] = []
    for batch_no, batch in enumerate(chunks(missing, batch_size), start=1):
        payload = fetch_batch(batch)
        if payload is None and len(batch) > 1:
            print(f"[Wikidata] Retrying failed batch {batch_no} as single-entity requests")
            failed_singletons: List[Sequence[str]] = []
            for singleton in ([qid] for qid in batch):
                payload = fetch_batch(singleton)
                if payload is None:
                    failed_singletons.append(singleton)
                    continue
                cache_entities(singleton, payload)
                done += 1
                atomic_write_json(cache_path, cache)
            failed_batches.extend(failed_singletons)
        elif payload is None:
            failed_batches.append(batch)
        else:
            cache_entities(batch, payload)
            done += len(batch)
            atomic_write_json(cache_path, cache)

        print(f"[Wikidata] {min(done, len(missing))}/{len(missing)} subjects cached")

    if failed_batches:
        atomic_write_json(cache_path, cache)
        failed_count = sum(len(batch) for batch in failed_batches)
        print(
            f"[WARNING] Wikidata skipped {failed_count} subjects after repeated throttling. "
            "Run again with --resume to refresh them later."
        )

    return cache


# ---------------------------------------------------------------------------
# Candidate lists
# ---------------------------------------------------------------------------

def surface_distance(a: str, b: str) -> Tuple[int, int]:
    return (
        abs(len(a.split()) - len(b.split())),
        abs(len(a) - len(b)),
    )


def make_candidate_list(
    pool: Mapping[str, str],
    excluded_qids: Set[str],
    excluded_labels: Set[str],
    target_label: str,
    max_candidates: int,
    seed_text: str,
) -> List[Dict[str, str]]:
    eligible = []
    for qid, label in pool.items():
        if qid in excluded_qids:
            continue
        if normalize_label(label) in excluded_labels:
            continue
        eligible.append((qid, label))

    if not eligible:
        return []

    # Keep some same-surface/granularity options and some random diversity.
    near = sorted(eligible, key=lambda x: surface_distance(target_label, x[1]))
    rng = random.Random(stable_seed(seed_text))
    shuffled = eligible[:]
    rng.shuffle(shuffled)

    want_near = max_candidates // 2
    selected = near[:want_near]
    seen = {q for q, _ in selected}

    for q, label in shuffled:
        if q in seen:
            continue
        selected.append((q, label))
        seen.add(q)
        if len(selected) >= max_candidates:
            break

    rng.shuffle(selected)
    return [{"qid": q, "label": label} for q, label in selected]


# ---------------------------------------------------------------------------
# OpenAI calls
# ---------------------------------------------------------------------------

def structured_call(
    client: OpenAI,
    model: str,
    instructions: str,
    input_obj: Mapping[str, Any],
    schema: Mapping[str, Any],
    tools: List[Dict[str, Any]] | None = None,
    max_retries: int = 5,
) -> Dict[str, Any]:
    last = None
    for k in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "instructions": instructions,
                "input": json.dumps(input_obj, ensure_ascii=False),
                "text": {"format": dict(schema)},
            }
            if tools:
                kwargs["tools"] = tools
            resp = client.responses.create(**kwargs)
            return json.loads(resp.output_text)
        except Exception as e:
            last = e
            time.sleep(min(2 ** k, 20))
    raise RuntimeError(f"OpenAI call failed: {last}")


def propose_candidates(
    client: OpenAI,
    model: str,
    item: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    pid = str(item["rel_entity"])

    instructions = """Select plausible controlled counterfactual objects for a factual-retrieval experiment.

You are given a true subject-relation-object fact and a list of candidate objects.
The candidate list has ALREADY been filtered so that known Wikidata values for this
subject-relation pair are removed.

Select up to five candidates, ranked best first.

A good candidate:
- has the correct semantic type and granularity for the relation;
- is plausible enough that a reader would not reject it as absurd on sight;
- is not simply an alias/equivalent of the true object;
- is not obviously another correct, historical, or disputed value;
- is similar in specificity to the true object when possible.

IMPORTANT: Do NOT demand proof of falsity here. This is a candidate-proposal stage.
A separate verifier will search for positive evidence that a proposed value might
actually be true. Prefer useful candidates over returning UNSUITABLE too readily.

You MUST select only QIDs from the supplied candidate list."""

    inp = {
        "subject": item.get("sub"),
        "subject_qid": item.get("sub_entity"),
        "subject_wikipedia": item.get("sub_wiki_url"),
        "relation": RELATION_NAMES.get(pid, item.get("rel")),
        "property": pid,
        "true_object": item.get("obj"),
        "true_object_qid": item.get("obj_entity"),
        "warning": RELATION_NOTES.get(pid, ""),
        "multi_valued_relation": pid in HIGH_RISK_MULTI_VALUED,
        "candidate_objects": list(candidates),
    }

    return structured_call(
        client, model, instructions, inp, GENERATION_FORMAT
    )


def adversarial_verify(
    client: OpenAI,
    model: str,
    item: Mapping[str, Any],
    candidate_qid: str,
    candidate_label: str,
    use_web: bool,
) -> Dict[str, Any]:
    pid = str(item["rel_entity"])

    instructions = """Perform an adversarial POSITIVE-EVIDENCE check for a candidate contradiction.

The candidate has already passed these automatic checks:
1. it is a known object type used with the same Wikidata relation elsewhere;
2. it is NOT listed among the subject's current Wikidata values for this relation.

Your task is NOT to prove a universal negative. Instead, actively look for reasons
the candidate should be REJECTED.

Reject if you find credible evidence that the candidate:
- is actually a true alternative value for this subject/relation;
- was historically true;
- is disputed/uncertain in a way that makes calling it false unsafe;
- has the wrong semantic type or granularity;
- is effectively an alias/equivalent/broader/narrower form of a correct value.

If, after checking, you find NO positive evidence that it is true, historical,
disputed, or type-incompatible, return PASS_NO_POSITIVE_EVIDENCE.

For multi-valued relations, search especially carefully for alternative true values.
Be concise in the reason."""

    inp = {
        "subject": item.get("sub"),
        "subject_qid": item.get("sub_entity"),
        "subject_wikipedia": item.get("sub_wiki_url"),
        "relation": RELATION_NAMES.get(pid, item.get("rel")),
        "property": pid,
        "true_object": item.get("obj"),
        "true_object_qid": item.get("obj_entity"),
        "candidate_object": candidate_label,
        "candidate_qid": candidate_qid,
        "warning": RELATION_NOTES.get(pid, ""),
    }

    tools = [{"type": "web_search"}] if use_web else None
    return structured_call(
        client, model, instructions, inp, VERIFY_FORMAT, tools=tools
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_context(pid: str, subject: str, obj: str) -> str:
    return RELATION_TEMPLATES[pid].format(s=subject, o=obj)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_one(
    qid: str,
    item: Mapping[str, Any],
    relation_pool: Mapping[str, Mapping[str, str]],
    dataset_truth: Mapping[Tuple[str, str], Set[str]],
    wikidata_truth: Mapping[str, Mapping[str, List[str]]],
    client: OpenAI,
    gen_model: str,
    verify_model: str,
    max_candidates: int,
    use_web: bool,
) -> Dict[str, Any]:
    out = dict(item)

    pid = str(item.get("rel_entity", "")).strip()
    sq = str(item.get("sub_entity", "")).strip()
    subject = str(item.get("sub", "")).strip()
    true_obj = str(item.get("obj", "")).strip()
    true_qid = str(item.get("obj_entity", "")).strip()

    if pid not in RELATION_TEMPLATES:
        out["contradiction"] = {
            "status": "unsupported_relation",
            "reason": pid,
        }
        return out

    known_true: Set[str] = set(dataset_truth.get((sq, pid), set()))
    known_true.add(true_qid)
    known_true.update(
        (wikidata_truth.get(sq, {}) or {}).get(pid, []) or []
    )

    excluded_labels = {
        normalize_label(true_obj),
        *(normalize_label(x) for x in (item.get("obj_aliases") or [])),
    }

    cand_list = make_candidate_list(
        relation_pool.get(pid, {}),
        excluded_qids=known_true,
        excluded_labels=excluded_labels,
        target_label=true_obj,
        max_candidates=max_candidates,
        seed_text=qid,
    )

    if not cand_list:
        out["contradiction"] = {
            "status": "no_candidates_after_filter",
            "known_true_qids": sorted(known_true),
        }
        return out

    proposal = propose_candidates(
        client, gen_model, item, cand_list
    )

    if proposal.get("status") != "CANDIDATES" or not proposal.get("candidates"):
        out["contradiction"] = {
            "status": "unsuitable_generation",
            "generation_reason": proposal.get("reason", ""),
            "known_true_qids": sorted(known_true),
        }
        return out

    supplied = {x["qid"]: x["label"] for x in cand_list}
    attempts: List[Dict[str, Any]] = []

    for ranked in proposal["candidates"]:
        cq = str(ranked.get("qid", "")).strip()
        if cq not in supplied:
            continue
        clabel = supplied[cq]

        # Final hard KG exclusion.
        if cq in known_true:
            attempts.append({
                "candidate_qid": cq,
                "candidate_label": clabel,
                "verification": "REJECT_KNOWN_TRUE",
            })
            continue

        verdict = adversarial_verify(
            client,
            verify_model,
            item,
            candidate_qid=cq,
            candidate_label=clabel,
            use_web=use_web,
        )

        attempts.append({
            "candidate_qid": cq,
            "candidate_label": clabel,
            "proposal_reason": ranked.get("reason", ""),
            "verification": verdict.get("label"),
            "verification_reason": verdict.get("reason", ""),
        })

        if verdict.get("label") != "PASS_NO_POSITIVE_EVIDENCE":
            continue

        out["contradiction"] = {
            "status": "accepted",
            "false_obj": clabel,
            "false_obj_entity": cq,
            "correct_context": render_context(pid, subject, true_obj),
            "contradictory_context": render_context(pid, subject, clabel),
            "automatic_checks": {
                "same_relation_object_pool": True,
                "not_known_true_in_input_or_wikidata": True,
                "not_true_object_alias": True,
            },
            "generation_model": gen_model,
            "verification_model": verify_model,
            "web_verification_enabled": use_web,
            "known_true_qids": sorted(known_true),
            "attempts": attempts,
        }
        return out

    out["contradiction"] = {
        "status": "unsuitable_after_verification",
        "known_true_qids": sorted(known_true),
        "generation_model": gen_model,
        "verification_model": verify_model,
        "web_verification_enabled": use_web,
        "attempts": attempts,
    }
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_summary(output: Mapping[str, Any]) -> None:
    statuses = Counter()
    verify = Counter()
    by_relation = defaultdict(Counter)

    for item in output.values():
        c = item.get("contradiction", {}) or {}
        st = c.get("status", "missing")
        statuses[st] += 1
        pid = str(item.get("rel_entity", "unknown"))
        by_relation[pid][st] += 1

        for a in c.get("attempts", []) or []:
            if a.get("verification"):
                verify[a["verification"]] += 1

    print("\n=== STATUS SUMMARY ===")
    for k, v in statuses.most_common():
        print(f"{k:35s} {v}")

    if verify:
        print("\n=== VERIFICATION SUMMARY ===")
        for k, v in verify.most_common():
            print(f"{k:35s} {v}")

    print("\n=== BY RELATION ===")
    for pid in sorted(by_relation):
        total = sum(by_relation[pid].values())
        acc = by_relation[pid].get("accepted", 0)
        name = RELATION_NAMES.get(pid, pid)
        print(f"{pid:6s} {name:18s} accepted={acc:5d}/{total:<5d} ({acc/total:.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ACCEPTED_STATUS = "accepted"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--wikidata-cache", type=Path, default=Path("wikidata_truth_cache.json"))
    p.add_argument("--gen-model", default=os.getenv("OPENAI_GEN_MODEL", "gpt-5.6-luna"))
    p.add_argument("--verify-model", default=os.getenv("OPENAI_VERIFY_MODEL", "gpt-5.6-terra"))
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-wikidata", action="store_true")
    p.add_argument("--no-web-verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    data = load_json(args.input)
    relation_pool, dataset_truth = build_indexes(data)

    if args.skip_wikidata:
        wd: Dict[str, Dict[str, List[str]]] = {}
        print("[WARNING] Wikidata exclusion disabled.")
    else:
        wd = preload_wikidata_truth(data, args.wikidata_cache)

    output = (
        read_json(args.output)
        if args.resume and args.output.exists()
        else {}
    )

    client = OpenAI()
    keys = list(data)
    if args.limit is not None:
        keys = keys[:args.limit]

    since_save = 0

    for i, qid in enumerate(keys, 1):
        old_status = (
            output.get(qid, {})
            .get("contradiction", {})
            .get("status")
        )
        if args.resume and old_status == ACCEPTED_STATUS:
            print(f"[{i}/{len(keys)}] {qid}: skip {old_status}")
            continue

        try:
            output[qid] = process_one(
                qid=qid,
                item=data[qid],
                relation_pool=relation_pool,
                dataset_truth=dataset_truth,
                wikidata_truth=wd,
                client=client,
                gen_model=args.gen_model,
                verify_model=args.verify_model,
                max_candidates=args.max_candidates,
                use_web=not args.no_web_verify,
            )
            status = output[qid]["contradiction"]["status"]
            print(f"[{i}/{len(keys)}] {qid}: {status}")
        except Exception as e:
            output[qid] = dict(data[qid])
            output[qid]["contradiction"] = {
                "status": "error",
                "error": repr(e),
            }
            print(f"[{i}/{len(keys)}] {qid}: ERROR {e}")

        since_save += 1
        if since_save >= args.save_every:
            atomic_write_json(args.output, output)
            since_save = 0

    atomic_write_json(args.output, output)
    print_summary(output)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
