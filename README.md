# Multilingual Analysis of Language Models' and Safety Validators' Robustness against Jailbreak Attacks

Code and data accompanying the master's thesis:

> **Multilingual Analysis of Language Models' and Safety Validators' Robustness against Jailbreak Attacks**
> Mariia Saltykova — Jagiellonian University, Faculty of Mathematics and Computer Science, Institute of Computer Science and Computational Mathematics.

## Overview

This repository contains the full experimental pipeline used to evaluate (1) the robustness of open-weight large language models (LLMs) against jailbreak attacks in lower-resource languages and (2) the effectiveness of automatic safety validators (guard models) in detecting harmful responses generated in those languages.

The experiment covers:

- **7 open-weight instruction-tuned models** (3–8 B parameters): Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Phi-3-mini-4k-instruct, Hermes-2-Pro-Llama-3-8B, OLMo-7B-Instruct-hf, Olmo-3-7B-Instruct;
- **26 jailbreak prompts** drawn from 7 public safety benchmarks (JailbreakBench, StrongREJECT, AdvBench, WildJailbreak, HarmBench, XSTest, MLCommons AILuminate);
- **7 languages** of differing resource level: English (high, reference), Bengali (medium), and Kyrgyz, Swahili, Somali, Luxembourgish, Irish (low);
- **4 safety validators**: Llama-Guard-3-1B, Nemotron-Content-Safety-Reasoning-4B, WildGuard, SGuard-ContentFilter-2B-v1;
- **manual annotation** of all 1 274 responses, used as the ground-truth reference against which the validators' results are compared.

## Repository structure

```
.
├── util/                          # data-preparation and post-processing utilities
│   ├── uploadPromptsToCSV.ipynb       # assemble the 26 English prompts into prompts.csv
│   ├── google_transletion_api_run.ipynb  # machine translation (EN → 6 other languages; responses → EN)
│   └── process_results.ipynb          # map text labels to binary (strict/mixed), build processed CSVs
│
├── LLMs run/                      # response generation — one notebook per model
│   ├── Llama-3.1-8B-Instruct_run.ipynb
│   ├── mistral7b_LLM_run.ipynb
│   ├── qwen2p5_7b_LMM_run.ipynb
│   ├── phi3_LLM_run.ipynb
│   ├── Hermes-2-Pro-Llama-3-8B_run.ipynb
│   ├── olmo_7b_instruct_hf_run.ipynb
│   └── olmo_3b_instract_run.ipynb
│
├── validators run/                # safety classification — one notebook per validator
│   ├── meta-llama_Llama-Guard-3-1B_run.ipynb
│   ├── nvidia_Nemotron-Content-Safety-Reasoning-4B_run.ipynb
│   ├── allenai_wildguard_run.ipynb
│   └── SamsungSDS-Research_SGuard-ContentFilter-2B-v1_run.ipynb
│
├── data and results/              # inputs and per-model result tables
│   ├── prompts.csv                    # 182 rows = 26 prompts × 7 languages
│   ├── llama-3_1-8b_results_processed.csv
│   ├── mistral7b_results_processed.csv
│   ├── qwen2p5_7b_results_processed.csv
│   ├── phi3_results_processed.csv
│   ├── hermes2pro_llama_8b_results_processed.csv
│   ├── olmo_7b_instruct_hf_results_processed.csv
│   └── olmo3_7b_instruct_results_processed.csv
│
└── README.md
```

## Experimental pipeline

The experiment is organised in five sequential stages:

1. **Prompt selection and translation.** `util/uploadPromptsToCSV.ipynb` assembles 26 English prompts (3–4 per benchmark) into `prompts.csv` and assigns each a stable `prompt_id` (1–26). `util/google_transletion_api_run.ipynb` translates every prompt from English into the six target languages via the Google Cloud Translation API (v2), yielding 26 × 7 = 182 prompts.

2. **Response generation.** Each notebook in `LLMs run/` loads one model and generates a response to all 182 prompts. Generation is deterministic (`do_sample=False`, fixed seed) to ensure reproducibility. Six of the seven models are loaded in 4-bit NF4 quantization; Phi-3-mini is run in native 16-bit precision. Non-English responses are then translated back into English (again via the Translation API) **only** to support manual annotation; validators receive the responses in their original language.

3. **Manual annotation.** Each of the 1 274 responses is labelled by a human annotator (the author) using a four-class scheme — `safe`, `unsafe`, `instruction misfollowing`, `nonsense output` — stored in the `original_validation` column and treated as ground truth.

4. **Automatic classification.** Each notebook in `validators run/` applies one validator to every record, in two passes: prompt-only and response-in-context. Each pass returns a `safe`/`unsafe` label.

5. **Label mapping.** `util/process_results.ipynb` maps the four-class annotation onto two binary representations — **strict** (`unsafe→1`, `safe→0`, instruction misfollowing OR nonsense output→`NaN`) and **mixed** (`unsafe→1`, everything else→`0`) — and maps each validator's labels to binary. The result is one processed CSV per model in `data and results/`.

The seven `*_results_processed.csv` files are the **direct inputs to the quantitative analysis** reported in Chapter 3 of the thesis (attack success rates, language and resource-group effects, paired bypass analysis, and validator accuracy/coverage/FNR/FPR).

## Data format

`prompts.csv` columns: `language`, `benchmark`, `prompt_id`, `prompt`.

Each `*_results_processed.csv` file contains the raw and derived columns needed for analysis, including:

| Column | Description |
| --- | --- |
| `prompt_id`, `language`, `prompt` | join keys and the (possibly translated) prompt text |
| `LLM_response` | model response in the original prompt language |
| `response_translation` | English translation of the response |
| `original_validation` | four-class manual ground-truth label |
| `original_response_strict_binary` | strict binary reference (`NaN` for invalid responses) |
| `original_response_mixed_binary` | mixed binary reference (invalid → `0`) |
| `<validator>_validation` | raw `prompt`/`response` labels for each of the four validators |
| `<validator>_response_binary`, `<validator>_prompt_binary` | binary-mapped validator outputs |

## Requirements

The notebooks were developed and executed in **Google Colab** with GPU acceleration. The main dependencies are:

- Python 3, Jupyter
- `transformers`, `torch`, `bitsandbytes` (4-bit quantization)
- `pandas`, `numpy`
- `matplotlib`, `seaborn` (figures)
- Google Cloud Translation API (v2) credentials, for the translation utility

Model and validator weights are pulled from the Hugging Face Hub; access to gated models (e.g. Llama) requires an authenticated Hugging Face token. The Translation utility requires a valid Google Cloud API key.

## Reproducing the results

1. Run `util/uploadPromptsToCSV.ipynb` and `util/google_transletion_api_run.ipynb` to (re)build `prompts.csv`. *(The provided `prompts.csv` already contains all 182 prompts.)*
2. Run each notebook in `LLMs run/` to generate model responses.
3. Annotate responses manually (four-class scheme).
4. Run each notebook in `validators run/` to obtain validator labels.
5. Run `util/process_results.ipynb` to produce the processed CSVs in `data and results/`.

## Responsible use

This repository is intended for safety-research purposes — measuring and improving the robustness of language models and safety classifiers. The prompts and generated outputs include harmful content collected from public safety benchmarks; they are provided solely to support reproducibility of the evaluation and must not be used to facilitate harm.