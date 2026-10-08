# Kaggle Qwen pilot review

Reviewed October 7, 2026. **Keep Qwen experimental. Original-query retrieval performed best in this pilot, but answer review found errors in both models.** No application defaults or README results were changed by this review.

## Evidence and scope

The downloaded `results/qwen_trial_results.zip` is preserved. Its six exported files are extracted under `results/qwen-trial-kaggle/`. All 50 unique paired questions match the seeded training sample (seed 42), and their gold labels match the local training dataset. The seven retrieval metrics were independently recomputed from the exported rankings and agree within 1e-10. Rankings have no duplicate IDs, and each top-5 ranking is the prefix of its top-10 ranking.

This is an exploratory training sample, not a full-corpus question evaluation or held-out quality estimate. All 50 paired answers were screened for document-ID mismatches and output caps; selected cases were manually inspected. **No aggregate semantic accuracy or human pass rate has been established.** Raw `answer_review.csv` remains available for a complete human review.

The corpus/index, Vietnamese bi-encoder, 25 dense + 25 sparse candidates, and BGE reranker were shared. Refined queries only affected retrieval. Both answer generators received the original question and the same top-5 evidence from original-query retrieval. The single-call refinement used here differs from the application's multi-call refinement pipeline.

## Verified retrieval results

Values are macro averages over 50 questions. Relevance uses exact gold article IDs; nDCG is binary. MAP follows the existing evaluator's denominator convention. Retrieval time excludes refinement and answer generation; one ranking is sliced at both cutoffs.

| Query | k | Precision | Recall | F1 | Hit rate | MAP | MRR | nDCG | Mean retrieval s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 5 | 0.1680 | 0.8100 | 0.2771 | 0.8200 | 0.6840 | 0.6940 | 0.7177 | 2.46 |
| Original | 10 | 0.0860 | 0.8300 | 0.1555 | 0.8400 | 0.6873 | 0.6973 | 0.7248 | 2.46 |
| Groq refinement | 5 | 0.1520 | 0.7400 | 0.2514 | 0.7600 | 0.6450 | 0.6600 | 0.6725 | 2.37 |
| Groq refinement | 10 | 0.0840 | 0.8100 | 0.1518 | 0.8200 | 0.6546 | 0.6667 | 0.6947 | 2.37 |
| Qwen refinement | 5 | 0.1480 | 0.7100 | 0.2438 | 0.7200 | 0.6390 | 0.6473 | 0.6591 | 2.40 |
| Qwen refinement | 10 | 0.0840 | 0.8100 | 0.1518 | 0.8200 | 0.6539 | 0.6622 | 0.6930 | 2.40 |

Relative to original-query Recall@5, Groq improved 1 question, worsened 5, and left 44 unchanged. Qwen improved none, worsened 5, and left 45 unchanged. Original top-10 adds only two percentage points of recall over top-5 in this sample; top-5 offers the better precision and smaller answer context.

One concrete refinement failure: case 33 asks about inspection stamps **expired less than one month ago**. Groq rewrites that as **about to expire, with less than one month remaining**, changing the legal situation despite the numeric-reference guard. Case numbers throughout this report are zero-based row indexes in `questions.jsonl`.

## Answer generation

| Model | Mean refinement s | Mean answer s | Reference-guard rejections | Rate-limit retries | Answers at output cap |
|---|---:|---:|---:|---:|---:|
| Groq gpt-oss-120b | 0.34 | 1.19 | 1 | 0 | 3/50 |
| Qwen 3.5 9B | 3.56 | 14.21 | 1 | 0 | 9/50 |

Qwen used Ollama 0.40.0, Q4_K_M quantization, thinking disabled, context 8192, and output cap 512 on a Tesla T4 runtime. Groq used low reasoning effort and completion cap 1024, including reasoning tokens. These budgets are unequal, and local GPU versus hosted API timing describes these deployments rather than intrinsic model speed.

Qwen cap cases: 5, 7, 10, 12, 15, 18, 20, 27, 43. Groq cap cases: 10, 18, 20. Inspection found abrupt endings, including partial sentences and identifiers. Finish reasons were not exported, so truncation is inferred from output usage and visible endings. Increasing the budget could improve completeness; it does not establish that citation substitutions would disappear.

### Citation screen and manually inspected examples

A deterministic screen normalizes Unicode/case/dashes, extracts document IDs matching `\d+/\d{4}/[\w-]+`, and flags answer IDs absent anywhere in the supplied context. It flags 19/50 Qwen answers and 1/50 Groq answers. **These are review flags, not hallucination rates.** Spelling, punctuation, and truncated IDs can trigger flags; matching IDs can still accompany incorrect document types, article numbers, or legal interpretations. It also cannot detect omitted requirements.

| Case | Supplied evidence | Inspected issue |
|---|---|---|
| 0 | `23/2018/TT-BGTVT`, Article 22 | Qwen substitutes `23/2018/QH14` and calls it the Railway Law. Groq preserves the ID but calls the circular a decree. |
| 6 | `15/2021/NĐ-CP`, Article 27(3), plus `05/2011/TT-BXD` | Both identify the investor; Qwen introduces `05/2011/QH-NST`. |
| 9 | `30/2020/NĐ-CP`, Article 7; `92/2012/TT-BQP`, Article 2 | Both answer that administrative documents include công điện. Qwen substitutes `92/2012/QĐ-TTg` and misnames the decree; Groq also labels the documents as laws. |
| 16 | `50/2019/TT-BTC` | Qwen gives the supported location-agreement answer but cites `50/2019/QĐ-TTg`. |
| 28 | `27/2018/TT-BNNPTNT` | Qwen substitutes `27/2018/QH14` and supplies an unsupported law title/date. |
| 32 | `24/2018/TT-BYT` and related health documents | Qwen substitutes `24/2018/QH14`. |
| 36 | Health circulars `45/2017/TT-BYT`, `55/2015/TT-BYT`, `08/2014/TT-BYT` | Qwen declines for insufficient evidence but rewrites the circular IDs as QH laws. |
| 41 | `01/2021/TT-NHNN` | Qwen substitutes `01/2021/QH14` and calls it the Law on Credit Institutions. |

Groq also makes substantive mistakes that this ID screen misses. In case 8, the evidence requires notification of defense counsel **plus** a lawyer/legal-aid/identity credential. Qwen preserves that distinction; Groq describes credentials as alternatives to the notification. In case 11, Groq extends a quoted exemption to a certificate by analogy without support in that clause. In case 3, Qwen more cautiously notes that the retrieved text does not directly state the requested training principles, while Groq presents related rules as principles.

The separate `results/qwen-trial-kaggle/answer_screen.csv` records all 100 answer screens, question IDs, cap flags, and selected manual notes. Its unreviewed fields do not imply a pass. The archived comparison and raw answer-review files are preserved.

## Recommendation for user review

Use original-query retrieval as the candidate default for the next evaluation. Keep 5/10 configurable. This pilot supports that direction but does not measure the application's existing multi-call refinement directly or justify replacing the historical full-sample README table.

Do not promote this Qwen configuration yet. A controlled follow-up should first improve citation fidelity and answer completeness, then review paired answers on the same evidence and a held-out sample. Groq also needs citation and legal-condition checks. Retrieval scores alone cannot approve either generator for legal answers.

Application changes and README replacement remain pending user review. No models were rerun during this import/review.
