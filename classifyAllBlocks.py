# author : Debanjan
# ================================================================
# KAGGLE SCRIPT: AWS BEDROCK LLAMA 3 70B - FULL PIPELINE
#
# This script performs two sequential tasks using one model:
# 1. TASK 1: Classifies all blocks from the PDF.
# 2. TASK 2: Filters for question/technical blocks and links them.
# ================================================================

import pandas as pd
import pickle
import time
import random
import re
import json
import warnings
import os
import boto3
from tqdm import tqdm
from botocore.exceptions import ClientError

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
# PART 1: SETUP AWS BEDROCK CLIENT (Done once)
# ================================================================
print("Setting up AWS Bedrock Client...")

# --- Configuration ---
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# --- Client Initialization ---
try:
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=BEDROCK_REGION,
        # Ensure AWS credentials are in environment variables or ~/.aws/credentials
    )
    print(f"✅ Bedrock client initialized for model: {BEDROCK_MODEL_ID}")
except Exception as e:
    print(f"❌ Failed to initialize Bedrock client: {e}")
    exit(1)

# --- Helper: Llama 3 Prompt Formatter ---
def format_llama3_prompt(user_message: str, system_message: str = "") -> str:
    """
    Wraps prompts in Llama 3 special tokens.
    """
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted

# --- Helper: Bedrock Call with Exponential Backoff ---
def invoke_bedrock_with_backoff(body, model_id, max_attempts=8):
    """
    Invokes AWS Bedrock model with exponential backoff for throttling errors.
    """
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            response = bedrock_client.invoke_model(
                body=json.dumps(body),
                modelId=model_id,
                contentType="application/json",
                accept="application/json"
            )
            return response
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                if attempt == max_attempts - 1:
                    print(f"❌ Max retries reached. Bedrock Throttled.")
                    raise
                
                sleep_time = delay + random.uniform(0, 0.5)
                # print(f"⚠️ Throttled. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= 2
            else:
                # If it's not a throttling error, raise it immediately
                print(f"❌ Bedrock ClientError: {e}")
                raise
        except Exception as e:
            print(f"❌ Unexpected Bedrock Error: {e}")
            raise

# ================================================================
# PART 2: TASK 1 - BLOCK CLASSIFICATION
# ================================================================
print("\n" + "="*60)
print("STARTING TASK 1: BLOCK CLASSIFICATION")
print("="*60)

# --- 2.1: Define Classification Prompt ---
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
def bedrock_classify(block_text):
    """
    Runs the Llama 3 model on Bedrock to classify a single text block.
    """
    user_content = f"Here is the text block to classify:\n\n{block_text}"
    final_prompt = format_llama3_prompt(user_content, system_prompt_classify)

    body = {
        "prompt": final_prompt,
        "max_gen_len": 40,
        "temperature": 0.1,
        "top_p": 0.9,
    }

    try:
        response = invoke_bedrock_with_backoff(body, BEDROCK_MODEL_ID)
        response_body = json.loads(response.get("body").read())
        return response_body.get("generation").strip()
    except Exception as e:
        # print(f"Error in classify call: {e}")
        return ""

# --- 2.3: Run Classification Loop ---
try:
    input_pkl_path = "export_for_kaggle/all_blocks_for_classification.pkl"
    df_all_blocks = pickle.load(open(input_pkl_path, "rb"))
    print(f"Loaded {len(df_all_blocks)} blocks from {input_pkl_path}")
except FileNotFoundError:
    print(f"ERROR: Could not find {input_pkl_path}")
    print("Please upload 'all_blocks_for_classification.pkl' and update the path.")
    # Create dummy data
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

# Using tqdm for progress tracking
for i, row in tqdm(df_all_blocks.iterrows(), total=len(df_all_blocks), desc="Task 1: Classifying blocks"):
    text = row["latex_content"]

    if not text.strip():
        classification_results.append("other")
        continue

    try:
        raw_output = bedrock_classify(text)
        
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

df_all_blocks["block_type"] = classification_results

# --- 2.4: Save Classification Results ---
df_all_blocks.to_csv("all_blocks_classified.csv", index=False)
pickle.dump(df_all_blocks, open("all_blocks_classified.pkl", "wb"))

print("✅ Task 1: Classification complete.")
print("Saved 'all_blocks_classified.csv' and 'all_blocks_classified.pkl'")
print("\nBlock type counts:")
print(df_all_blocks["block_type"].value_counts())


# ================================================================
# PART 3: TASK 2 - QUESTION LINKING
# ================================================================
print("\n" + "="*60)
print("STARTING TASK 2: QUESTION LINKING")
print("="*60)

# --- 3.1: Load and Filter Data (from Task 1) ---
try:
    df_classified = pickle.load(open("all_blocks_classified.pkl", "rb"))
    
    # Filter for only relevant blocks
    df_filtered = df_classified[
        df_classified['block_type'].isin(['question', 'technical'])
    ].copy()
    
    # Reset index
    df = df_filtered.reset_index(drop=True)
    
    print(f"Loaded {len(df_classified)} classified blocks.")
    print(f"Filtered down to {len(df)} 'question' and 'technical' blocks for linking.")

except FileNotFoundError:
    print("Error: 'all_blocks_classified.pkl' not found. Cannot proceed to Task 2.")
    df = pd.DataFrame() 

if not df.empty:
    # --- 3.2: Define Linking Functions ---

    # STEP 1 — RAW CLASSIFIER
    def bedrock_raw_decision(history_blocks, candidate_block):
        hist_text = ""
        for i, b in enumerate(history_blocks):
            hist_text += f"Block {i+1}:\n{b}\n\n"

        prompt_text = f"""
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
        final_prompt = format_llama3_prompt(prompt_text)
        
        body = {
            "prompt": final_prompt,
            "max_gen_len": 200,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        
        try:
            response = invoke_bedrock_with_backoff(body, BEDROCK_MODEL_ID)
            response_body = json.loads(response.get("body").read())
            return response_body.get("generation").strip()
        except Exception as e:
            # print(f"Error in raw decision: {e}")
            return ""

    # STEP 2 — SUMMARIZER
    def bedrock_summarize(explanation_only):
        
        prompt_text = f"""
The text below is an explanation written by another AI:

EXPLANATION:
{explanation_only}

Your task:
Determine which ONE WORD classification that AI *intended*.

The previous model might have hallucinated and wrote randomly NEW or CONT here and there but the overall explanation would be correct. Infer intelligently from the explanation ignoring random insertions of NEW or CONT.
If the explanation starts off saying the block is continuation of the previous block, it is strongly continuation (CONT), the explanation might go into hallucination on the later part and say that it's new but it's actually continuation.


Return ONLY ONE WORD:
NEW     -> means block starts a new question
CONT    -> means block continues previous question

Do NOT re-evaluate the PDF blocks. Only infer from explanation.
"""
        final_prompt = format_llama3_prompt(prompt_text)

        body = {
            "prompt": final_prompt,
            "max_gen_len": 50,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        try:
            response = invoke_bedrock_with_backoff(body, BEDROCK_MODEL_ID)
            response_body = json.loads(response.get("body").read())
            
            raw_summary = response_body.get("generation").strip()
            ans = raw_summary.upper().strip()
            
            # Debug prints
            print('#########################THIS IS THE RAW SUMMARY OF STEP 2', end= " ")
            print("#########################", ans)
            
            if "NEW" in ans:
                return "NEW", raw_summary
            if "CONT" in ans:
                return "CONT", raw_summary
            return "CONT", raw_summary 
            
        except Exception as e:
            # print(f"Error in summarize: {e}")
            return "CONT", ""

    # --- 3.3: Run Linking Pipeline ---
    results = []
    history = []
    
    for i in tqdm(range(len(df)), desc="Task 2: Linking questions"):
        block = df.loc[i, "latex_content"]

        if i == 0:
            results.append("start of a new question")
            history.append(block)
            continue

        window = history[-4:]

        print("\n\n==============================================================")
        print(f"PROCESSING FILTERED BLOCK {i} (Original ID: {df.loc[i, 'id']})")
        print("==============================================================")

        try:
            # ----------------- STEP 1: RAW CLASSIFIER --------------------------
            raw1 = bedrock_raw_decision(window, block)

            print("\n-------------------- STEP 1 : FULL RAW OUTPUT --------------------")
            print(raw1)
            print("-------------------- END STEP 1 RAW -------------------------------\n")

            explanation = raw1.strip()
            
            print("\n-------------- EXPLANATION PASSED TO STEP 2 ----------------")
            print(explanation)
            print("-------------- END EXPLANATION -----------------------------\n")

            # ----------------- STEP 2: SUMMARIZER ------------------------------
            final_label, raw2 = bedrock_summarize(explanation)
            
            print("\n-------------------- STEP 2 : FULL SUMMARIZER OUTPUT --------------------")
            print(raw2)
            print("-------------------- END STEP 2 RAW -------------------------------\n")
            
            print("FINAL DECISION :", final_label)

            if final_label == "NEW":
                results.append("start of a new question")
            else:
                results.append("continuation of previous question")
        
        except Exception as e:
            print(f"Error linking block {i} (Original ID: {df.loc[i, 'id']}): {e}")
            results.append("continuation of previous question") # Default to CONT

        history.append(block)

    # --- 3.4: Save Final Linking Results ---
    df["question_start_type"] = results
    
    df.to_csv("qwen_debug_full_outputs.csv", index=False)
    pickle.dump(df, open("qwen_debug_full_outputs.pkl", "wb"))

    print("\n✅ Task 2: Question linking complete.")
    print("Saved 'qwen_debug_full_outputs.csv' and 'qwen_debug_full_outputs.pkl'")
else:
    print("Skipping Task 2 because no 'question' or 'technical' blocks were found.")
