import os
import sys
import json
import re
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# --- Add parent directory to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from llm_handler import get_llm_handler

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# CSV for BM25 mapping results: query_sha / query_msg / matched_msg / rank (required)
INPUT_CSV_PATH = 'bm25_top_10_mapping_results_commit_f.csv'

# CSV with draft messages to be refined: query_sha / original_message (required)
QUERY_INPUTS_CSV = 'refinement_inputs.csv'

# K values to test (few-shot: 1,3,5,10 / zero-shot: 0)
K_VALUES_TO_TEST = [0, 1, 2, 3, 4, 5, 10]

# Model key
MODEL_KEY = "llama_7b"
# Batch size
BATCH_SIZE = 8

# Decoding parameters (optimized for quality)
DECODE_KW = dict(
    max_new_tokens=64,  # Increased to prevent message truncation
    num_beams=4,
    do_sample=False,
    repetition_penalty=1.05,
)

# Output directory + timestamp (freezes files per run)
RUN_ID = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = "runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# --- NEW: Refinement-Focused Prompts (in English) ---
# ==============================================================================

def build_refinement_few_shot_prompt(style_example: str, draft_message: str) -> str:
    """
    Few-shot prompt for the refinement task.
    It explicitly asks the LLM to rewrite a draft message using a style reference.
    """
    return f"""You are an expert technical editor who refines draft Git commit messages to improve their quality.
Your task is to rewrite the [DRAFT MESSAGE] to be more clear, concise, and professional, strictly following the style of the [STYLE REFERENCE].

Rules:
- Start with a conventional commit type (e.g., feat, fix, refactor, docs).
- Keep the core meaning of the draft.
- Do not add any explanations; output only the single-line refined message.

---
[STYLE REFERENCE]:
{style_example}

---
[DRAFT MESSAGE TO REFINE]:
{draft_message or "(empty message)"}

---
[REFINED MESSAGE]:
"""

def build_refinement_zero_shot_prompt(draft_message: str) -> str:
    """
    Zero-shot prompt for the refinement task.
    """
    return f"""You are an expert technical editor who refines draft Git commit messages to improve their quality.
Your task is to rewrite the [DRAFT MESSAGE] below into a high-quality, single-line commit message.

Rules:
- Start with a conventional commit type (e.g., feat, fix, refactor, docs).
- Use the imperative mood (e.g., "add feature" not "added feature").
- Keep the core meaning of the draft.
- Output only the single-line refined message without any explanations.

---
[DRAFT MESSAGE TO REFINE]:
{draft_message or "(empty message)"}

---
[REFINED MESSAGE]:
"""

# ==============================================================================
# --- Helper Functions (Unchanged) ---
# ==============================================================================

def _safe_pick_index(text, k):
    if text is None: return None
    m = re.search(r'\d+', str(text))
    if not m: return None
    try:
        idx = int(m.group(0)) - 1
        return idx if 0 <= idx < k else None
    except Exception:
        return None

def call_generate_batch(handler, prompts, **kwargs):
    try:
        return handler.generate_batch(prompts, **kwargs)
    except TypeError:
        return handler.generate_batch(prompts)

def postprocess_line(s: str):
    if not s: return s
    s = s.strip().splitlines()[0].strip()
    s = re.sub(r"[.]\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================
def main():
    """Main execution function"""
    # Step 0: Load draft messages for refinement
    draft_messages = {}
    if os.path.exists(QUERY_INPUTS_CSV):
        try:
            qdf = pd.read_csv(QUERY_INPUTS_CSV)
            # Read 'original_message' column for the refinement task
            draft_messages = dict(zip(qdf['query_sha'], qdf['original_message']))
            print(f"[INFO] Loaded {len(draft_messages)} draft messages from {QUERY_INPUTS_CSV}")
        except Exception as e:
            print(f"[ERROR] Failed to read {QUERY_INPUTS_CSV}. Make sure it has 'query_sha' and 'original_message' columns. Details: {e}")
            return
    else:
        print(f"[ERROR] Input file not found: {QUERY_INPUTS_CSV}")
        return

    print(f"Step 1: Loading and grouping BM25 mapping data from '{INPUT_CSV_PATH}'...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found. Please check the path: '{INPUT_CSV_PATH}'")
        return

    grouped = df.groupby('query_sha', sort=False)
    experiments_data = []
    for query_sha, group in tqdm(grouped, desc="Processing data"):
        group = group.sort_values(by='rank', kind='mergesort')
        candidates = [row.to_dict() for _, row in group.iterrows()]
        experiments_data.append({
            'query_sha': query_sha,
            'original_query_msg': group.iloc[0]['query_msg'], # Gold message for evaluation
            'draft_message': draft_messages.get(query_sha),   # The message to be refined
            'candidates': candidates
        })
    print(f"Successfully created {len(experiments_data)} query groups.")

    print(f"\nStep 2: Initializing LLM handler for model: '{MODEL_KEY}'...")
    llm_handler = get_llm_handler(MODEL_KEY)
    print("LLM handler initialized successfully.")

    for k in K_VALUES_TO_TEST:
        out_path = os.path.join(OUTPUT_DIR, f"{MODEL_KEY}_k{k}_{RUN_ID}.jsonl")
        print(f"\nStep 3: Starting the main experiment loop for K={k} -> {out_path}")
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for i in tqdm(range(0, len(experiments_data), BATCH_SIZE), desc=f"K={k}"):
                batch_data = experiments_data[i:i+BATCH_SIZE]

                # --- Agent A: Select the best style reference (for k > 1) ---
                if k > 1:
                    prompts_a = []
                    for data_item in batch_data:
                        candidates_for_a = data_item['candidates'][:k]
                        prompt = (
                            "You are an expert in analyzing the semantics of commit messages.\n"
                            "From the [CANDIDATES], choose the one whose style and intent best match the [DRAFT MESSAGE].\n"
                            "Provide only the number of the best choice without any other text.\n\n"
                            f"[DRAFT MESSAGE]:\n{data_item.get('draft_message') or '(no message)'}\n\n"
                            "[CANDIDATES]:\n" + "\n".join(
                                f"{j + 1}. {c['matched_msg']}" for j, c in enumerate(candidates_for_a)
                            )
                        )
                        prompts_a.append(prompt)

                    responses_a = call_generate_batch(llm_handler, prompts_a, **DECODE_KW)

                    chosen_examples = []
                    for idx, resp_text in enumerate(responses_a):
                        try:
                            chosen_idx = _safe_pick_index(resp_text, k)
                            if chosen_idx is None: raise ValueError
                            chosen_examples.append(batch_data[idx]['candidates'][chosen_idx]['matched_msg'])
                        except Exception:
                            chosen_examples.append(batch_data[idx]['candidates'][0]['matched_msg']) # Fallback to rank 1
                
                elif k == 1:
                    chosen_examples = [item['candidates'][0]['matched_msg'] for item in batch_data]
                else: # k == 0
                    chosen_examples = [None for _ in batch_data]

                # --- Agent B: Refine the draft message ---
                prompts_b = []
                for idx, data_item in enumerate(batch_data):
                    style_example = chosen_examples[idx]
                    draft_to_refine = data_item.get('draft_message')
                    
                    if style_example is None: # zero-shot
                        prompt = build_refinement_zero_shot_prompt(draft_to_refine)
                    else: # few-shot
                        prompt = build_refinement_few_shot_prompt(style_example, draft_to_refine)
                    prompts_b.append(prompt)

                generated_commits = call_generate_batch(llm_handler, prompts_b, **DECODE_KW)
                generated_commits = [postprocess_line(x) for x in generated_commits]

                # --- Save results ---
                for idx, data_item in enumerate(batch_data):
                    result = {
                        'query_sha': data_item['query_sha'],
                        'k_value': k,
                        'original_query_msg': data_item['original_query_msg'], # Gold message
                        'draft_message': data_item.get('draft_message'),      # The input draft
                        'selected_style_example': chosen_examples[idx],
                        'generated_commit': generated_commits[idx]            # The refined output
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\nStep 4: K={k} finished. Results saved to '{out_path}'")

if __name__ == "__main__":
    main()