# Figure 3. Judging prompts

## Correctness judge

System message:

```text
You are a strict evaluator. Output only true or false.
```

User message:

```text
Question: {question}
Answer: {ground_truth}
Candidate: {candidate_answer}
Is candidate correct?
```

## Contradictory-context outcome judge

System message:

```text
You are a strict evaluator. Output only one label: TRUE_OBJECT, FALSE_CONTEXT_OBJECT, or OTHER.
```

User message:

```text
Question: {question}
True object: {true_object}
False context object: {false_context_object}
Candidate: {candidate_answer}
Classify the candidate answer.
Return only TRUE_OBJECT if the candidate answer is the true object.
Return only FALSE_CONTEXT_OBJECT if the candidate answer is the false context object.
Return only OTHER otherwise.
```
