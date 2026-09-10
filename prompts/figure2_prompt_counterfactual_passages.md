# Figure 2. Counterfactual-passage prompts

## Candidate proposal

```text
Select plausible controlled counterfactual objects for a factual-retrieval experiment.

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

You MUST select only QIDs from the supplied candidate list.
```

Input fields:

```text
subject, subject_qid, subject_wikipedia, relation, property,
true_object, true_object_qid, warning, multi_valued_relation,
candidate_objects
```

## Adversarial verification

```text
Perform an adversarial POSITIVE-EVIDENCE check for a candidate contradiction.

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
Be concise in the reason.
```

Input fields:

```text
subject, subject_qid, subject_wikipedia, relation, property,
true_object, true_object_qid, candidate_object, candidate_qid, warning
```
