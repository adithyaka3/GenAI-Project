# author : Debanjan
# ================================================================
# KAGGLE SCRIPT: QWEN 7B - FULL PIPELINE
#
# This script performs two sequential tasks using one model:
# 1. TASK 1: Classifies all blocks from the PDF.
# 2. TASK 2: Filters for question/technical blocks and links them.
# ================================================================

import pandas as pd
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re
import json
import gc
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using `max_length`*")

# ================================================================
# PART 1: LOAD MODEL (Done once)
# ================================================================
model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading 4-bit quantized model: {model_name}")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 (best quality)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # Extra compression
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   # ← enable quantization
    device_map="auto",
)

print("✅ 4-bit quantized model loaded successfully.")

# ================================================================
# PART 2: TASK 1 - BLOCK CLASSIFICATION
# ================================================================
print("\n" + "="*60)
print("STARTING TASK 1: BLOCK CLASSIFICATION")
print("="*60)

# --- 2.1: Define Classification Prompt and Labels ---
# This is the prompt from `classifier.py`
system_prompt_classify = """
You are an expert document classifier. Your task is to classify a block of text from an assignment PDF into ONE of the following six categories.
Respond with ONLY a JSON object in the format {"category": "category_name"}, and nothing else. Do not explain your reasoning.

Here are the categories and their definitions:

1.  **instruction**: This is for document-level instructions. (e.g., "Instructions: For all the questions, write your own functions...", "Submission guidelines...")
2.  **metadata**: This is for top-level information about the document itself. (e.g., "Due Date: October 26, 2025...", "Course: E9 241...", "Total Marks...")
3.  **note**: This is a small, specific note or hint for a *particular* question. It often starts with "**Note:**" or "**Hint:**".
4.  **question**: This is the main text of a *new* question or a major sub-section. It describes the *problem* to be solved. (e.g., "1. Directional Filtering:", "Question 2: Image Restoration")
5.  **technical**: This provides supporting details *for* an assignment question. This includes lists of parameters, equations, or descriptive text *after* a question title. (e.g., "Directional filtering is used to emphasize...", "Compute the 2D DFT...")
6.  **other**: Any text that does not fit, such as a "References" section, page headers/footers, or junk text.

Your response must be *only* the JSON object.
"""
labels_for_classifier = ["instruction", "metadata", "note", "question", "technical", "other"]

# --- 2.2: Define Classification Function ---
def qwen_classify(block_text):
    """
    Runs the Qwen model to classify a single text block.
    """
    chat = [
        {"role": "user", "content": f"{system_prompt_classify}\n\nHere is the text block to classify:\n\n{text}"}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    return raw_output

# --- 2.3: Run Classification Loop ---
try:
    # **IMPORTANT**: Update this path to your Kaggle input directory
    input_pkl_path = "export_for_kaggle/all_blocks_for_classification.pkl"
    df_all_blocks = pickle.load(open(input_pkl_path, "rb"))
    print(f"Loaded {len(df_all_blocks)} blocks from {input_pkl_path}")
except FileNotFoundError:
    print(f"ERROR: Could not find {input_pkl_path}")
    print("Please upload 'all_blocks_for_classification.pkl' and update the path.")
    # Create dummy data to allow script to continue for testing
    df_all_blocks = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "latex_content": [
            "**Instructions:** ...",
            "**Note:** ...",
            "1. What is 2+2?",
            "Explain your answer."
        ]
    })

classification_results = []

print("Clearing GPU cache before classification...")
gc.collect()
torch.cuda.empty_cache()

for i, row in tqdm(df_all_blocks.iterrows(), total=len(df_all_blocks), desc="Task 1: Classifying blocks"):
    text = row["latex_content"]

    if not text.strip():
        classification_results.append("other")
        continue

    try:
        raw_output = qwen_classify(text)
        
        # --- JSON PARSING LOGIC ---
        found_label = "other" # Default
        json_match = re.search(r'\{.*\}', raw_output)
        
        if json_match:
            try:
                json_string = json_match.group(0).replace("'", "\"")
                data = json.loads(json_string)
                if "category" in data and data["category"] in labels_for_classifier:
                    found_label = data["category"]
            except json.JSONDecodeError:
                pass # Fallback
        
        if found_label == "other": # Fallback to string matching
            for label in labels_for_classifier:
                if label in raw_output.lower():
                    found_label = label
                    break
        
        classification_results.append(found_label)

    except Exception as e:
        print(f"Error classifying block {row['id']}: {e}")
        classification_results.append("other")
        gc.collect()
        torch.cuda.empty_cache()

df_all_blocks["block_type"] = classification_results

# --- 2.4: Save Classification Results ---
df_all_blocks.to_csv("all_blocks_classified.csv", index=False)
pickle.dump(df_all_blocks, open("all_blocks_classified.pkl", "wb"))

print("✅ Task 1: Classification complete.")
print("Saved 'all_blocks_classified.csv' and 'all_blocks_classified.pkl'")
print("\nBlock type counts:")
print(df_all_blocks["block_type"].value_counts())

import pandas as pd
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import json
import gc
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using `max_length`*")

# ================================================================
# PART 3: TASK 2 - QUESTION LINKING
# ================================================================
print("\n" + "="*60)
print("STARTING TASK 2: QUESTION LINKING")
print("="*60)

# --- 3.0: Clear GPU Cache (as requested) ---
print("Clearing GPU cache before linking...")
gc.collect()
torch.cuda.empty_cache()

# --- 3.1: Load and Filter Data (from Task 1) ---
try:
    # This file was created by the classification task
    df_classified = pickle.load(open("all_blocks_classified.pkl", "rb"))
    
    # Filter for only relevant blocks (as you confirmed)
    # We treat 'question' and 'technical' as the same for linking
    df_filtered = df_classified[
        df_classified['block_type'].isin(['question', 'technical'])
    ].copy()
    
    # Reset index so the sequential loop (for i in range(len(df))) works
    df = df_filtered.reset_index(drop=True)
    
    print(f"Loaded {len(df_classified)} classified blocks.")
    print(f"Filtered down to {len(df)} 'question' and 'technical' blocks for linking.")

except FileNotFoundError:
    print("Error: 'all_blocks_classified.pkl' not found. Cannot proceed to Task 2.")
    print("Please ensure Task 1 (Classification) ran successfully.")
    df = pd.DataFrame() # Create empty df

if not df.empty:
    # --- 3.2: Define Linking Functions ---
    # We define these here so this code block is self-contained
    # Note: 'model' and 'tokenizer' are assumed to be loaded from Part 1

    # STEP 1 — RAW CLASSIFIER (USER'S PREFERRED PROMPT)
    def qwen_raw_decision(history_blocks, candidate_block):

        hist_text = ""
        for i, b in enumerate(history_blocks):
            hist_text += f"Block {i+1}:\n{b}\n\n"

        # --- THIS IS THE PROMPT YOU PROVIDED ---
        prompt = f"""
You are analyzing a sequence of assignment PDF blocks.

Below are up to the last 4 previous blocks:

{hist_text}

FINAL BLOCK (candidate):
{candidate_block}

Task:
Analyze the FINAL BLOCK in the context of the previous blocks.
Decide if it starts a NEW QUESTION or CONTINUES the previous one.

Cues you can look for but not limited to:
- NEW: The block introducing a new, distinct problem or section. This is almost always marked by a new question number (e.g., "1.", "2.", "Question 3:") or a new bold-faced title have more chance of being a new question. 
- CONT: The block provides more details, explanation, instructions, or sub-parts for the *current* question or blocks that are just plain text, or start with "Note:", "Hint:", or are part of a numbered list that continues from the previous block, have more chance of being  continuations. If the block is a Note, then most chance is that it tells some specific things about the previous block. You can also check if the Note or Hint has some phrases or expressions similar to the previous block.

One strong cue you can look for start of a new question is that its previous block might have some marks written at the end (although this might happen for sub-parts also, so judge yourself).

Does the FINAL BLOCK start a NEW QUESTION (NEW)
or does it CONTINUE the same question (CONT)?

Explain your reasoning. 
"""

        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=4096).to("cuda")
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            # Using 200 tokens as in your script
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        generated_tokens = out[0][input_length:]
        raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return raw

    # STEP 2 — SUMMARIZER (USER'S PREFERRED PROMPT)
    def qwen_summarize(explanation_only):
        
        # --- THIS IS THE PROMPT YOU PROVIDED ---
        prompt = f"""
The text below is an explanation written by another AI:

EXPLANATION:
{explanation_only}

Your task:
Determine which ONE WORD classification that AI *intended*.

The previous model might have hallucinated and wrote randomly NEW or CONT here and there but the overall explanation would be correct. Infer intelligently from the explanation ignoring random insertions of NEW or CONT.
If the explanation starts off saying the block is continuation of the previous block, it is strongly continuation (CONT), the explanation might go into hallucination on the later part and say that it's new but it's actually continuation.


Return ONLY ONE WORD:
NEW     -> means block starts a new question
CONT   -> means block continues previous question

Do NOT re-evaluate the PDF blocks. Only infer from explanation.
"""

        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=4096).to("cuda")
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        generated_tokens = out[0][input_length:]
        raw_summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        ans = raw_summary.upper().strip()
        
        # Adding the debug print from your script
        print('#########################THIS IS THE RAW SUMMARY OF STEP 2', end= " ")
        print("#########################", ans)
        
        # Using the exact logic from your script
        if "NEW" in ans:
            return "NEW", raw_summary
        if "CONT" in ans:
            return "CONT", raw_summary
        return "CONT", raw_summary # Default fallback

    # --- 3.3: Run Linking Pipeline ---
    results = []
    history = []
    
    # We iterate through the *filtered* dataframe
    for i in tqdm(range(len(df)), desc="Task 2: Linking questions"):
        # `df.loc[i]` gives us the sequentially-ordered 'question'/'technical' blocks
        block = df.loc[i, "latex_content"]

        if i == 0:
            # The first block in this filtered set is always the start of the first question
            results.append("start of a new question")
            history.append(block)
            continue

        # Get the last 4 blocks *from the history of linked blocks*
        window = history[-4:]

        print("\n\n==============================================================")
        # We print the original 'id' from the PDF for easy debugging
        print(f"PROCESSING FILTERED BLOCK {i} (Original ID: {df.loc[i, 'id']})")
        print("==============================================================")

        try:
            # ----------------- STEP 1: RAW CLASSIFIER --------------------------
            raw1 = qwen_raw_decision(window, block)

            # --- ADDING FULL DEBUG PRINTS ---
            print("\n-------------------- STEP 1 : FULL RAW OUTPUT --------------------")
            print(raw1)
            print("-------------------- END STEP 1 RAW -------------------------------\n")

            explanation = raw1.strip()
            
            print("\n-------------- EXPLANATION PASSED TO STEP 2 ----------------")
            print(explanation)
            print("-------------- END EXPLANATION -----------------------------\n")

            # ----------------- STEP 2: SUMMARIZER ------------------------------
            final_label, raw2 = qwen_summarize(explanation)
            
            print("\n-------------------- STEP 2 : FULL SUMMARIZER OUTPUT --------------------")
            print(raw2)
            print("-------------------- END STEP 2 RAW -------------------------------\n")
            # --- END OF ADDED PRINTS ---
            
            print("FINAL DECISION :", final_label)

            if final_label == "NEW":
                results.append("start of a new question")
            else:
                results.append("continuation of previous question")
        
        except Exception as e:
            print(f"Error linking block {i} (Original ID: {df.loc[i, 'id']}): {e}")
            results.append("continuation of previous question") # Default to CONT on error
            gc.collect()
            torch.cuda.empty_cache()

        history.append(block)

    # --- 3.4: Save Final Linking Results ---
    # The 'results' list now aligns with the filtered 'df'
    df["question_start_type"] = results
    
    # This CSV/Pickle contains *only* the 'question'/'technical' blocks
    # but now with the 'question_start_type' column added.
    df.to_csv("qwen_debug_full_outputs.csv", index=False)
    pickle.dump(df, open("qwen_debug_full_outputs.pkl", "wb"))

    print("\n✅ Task 2: Question linking complete.")
    print("Saved 'qwen_debug_full_outputs.csv' and 'qwen_debug_full_outputs.pkl'")
else:
    print("Skipping Task 2 because no 'question' or 'technical' blocks were found.")