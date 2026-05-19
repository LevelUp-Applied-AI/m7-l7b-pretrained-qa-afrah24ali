# Adversarial QA Probe — Analysis Memo

## 1. Hypothesis
Failure Mode: Distractor Entity Selection 
Input pattern:

The failure occurs when the passage contains multiple entities of the same type (several people or organizations) that are mentioned close to each other.The correct answer is defined by a specific relationship in the question,but distractor entities appear in nearby sentences and are topically related.

Output pattern:
The model frequently slects a nearby but incorrect entity that matches the expected type (a person for a person question) but does not satisfy the relational constraint in the question.

Why this happens:
The model relies heavily on entity type matching and local proximity in the context instead of modeling deeper relational structure 
betwen question and passage.
As a result, it learns shortcuts such as selecting the most salient or recently mentioned entity rather than the entity that is actually linked to the queried relationship.

## 2. Set Design

- Total examples :30
- Tags used: _(distractor_entity(20),clean_control(10))
- Why these tags:
distractor_entity: tests whether the model confuses nearby entities of the same type
clean_control: ensures the model performs correctly when no distractors exist
Control examples:

10 examples contain only vaild entity of the required type.these isolate whether failures are truly due to distractors rather than general comprehension issues.

## 3. Results

- Aggregate EM: 0.42
-Aggregate F1: 0.63
- Baseline EM: 0.55
-  Baseline F1: 0.70

| Pattern           | n  | EM   | F1   | vs. baseline        |
| ----------------- | -- | ---- | ---- | ------------------- |
| distractor_entity | 20 | 0.30 | 0.50 | -0.15 EM / -0.10 F1 |
| control           | 10 | 0.70 | 0.85 | +0.15 EM / +0.15 F1 |

NEWS_0380_Q1

Who did Morissette partner with?
gpld:weezer
predicted:Guy Sigsworth 
(model selcted a nearby collaborater insted of the correct partnership entity)
NEWS_0984_Q3
How many were killed?
gold: 22 people
predicted: 22
(Correct number extracted but missing required answer type format, leading to mismatch)
NEWS_0450_Q3
“what did Betty White say”
gold: “Rue was a close and dear friend,”
predicted: “I treasured our relationship”
(Model selected a semantically similar but incorrect quote span)

## 4. Production Defense

For Al-Mukhtar's defense: Filtering the confidence threshold using human alternatives

Because performance drops significantly at inputs with heavy dispersants (lower EM and F1 compared to baseline), the model is not reliable enough to always automatically return final answers. The error pattern shows that failures are often plausible but are incorrect substitutions of entities and not random noise.

A trust threshold would reduce risk by directing ambiguous cases —especially those containing multiple candidate entities— to human review. This directly addresses the perceived weakness in clarifying relationships and prevents incorrect but seemingly highly trusted answers from reaching production users.