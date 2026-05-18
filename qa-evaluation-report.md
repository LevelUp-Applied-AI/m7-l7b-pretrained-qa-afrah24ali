# QA Evaluation Report

## Dataset description

This evaluation uses the curated tech/entertaiment slice`glnmario/news-qa-summarization`, packaged as extractive question-answering examples in`data/tech_news_qa.csv`.Each row contains a question, a new context, and a gold answer span that appears in the context.

## Model

Model: `distilbert-base-cased-distilled-squad`  
Hugging Face Hub: https://huggingface.co/distilbert-base-cased-distilled-squad

## Aggregate metrics

Exact Match: 0.34  
Token-F1: 0.46

Token-F1 is higher than EM which suggest the mpdel often finds partially overlapping answer spans but misses exact boundaries or chooses a related pharse. The gap is meaningful: the system has some extractive signal, but exact answer reliability is not strong on this news slice.

## Failure-mode taxonomy

1. Distractor entity selection. The model sometimes chooses a nearby person, title, or entity from the same article instead of the entity requested by the question. Example: `NEWS_0380_Q1`, question: "Who did Morissette partner with?", gold: `Weezer`, predicted: `Guy Sigsworth`.

2. Answer type mismatch. The model may return a plausible numeric or named span while the gold answer expects a different answer type, producing zero overlap. Example: `NEWS_0984_Q3`, question: "How many were killed?", gold: `people`, predicted: `22`.

3. Quote/span boundary confusion. For quote questions, the model can select a neighboring quotation fragment rather than the exact quoted statement in the gold span. Example: `NEWS_0450_Q3`, question: "what did betty white say", gold: `"Rue was a close and dear friend,"`, predicted: `"I treasured our relationship`.

## Domain judgment

I would not ship this model as-is for legal contract QA. The EM/F1 results show too many boundary and distractor errors, and legal workflows need stronger faithfulness, calibration, and preferably no-answer support when the requested fact is absent. It could be useful as an internal search assistant with human review, but not as an authoritative production answerer.