import os
import re
import json
import time
import random
import boto3
from typing import Dict, Optional
from pydantic import BaseModel, Field

# ==========================================
# 1. AWS Bedrock Client Setup
# ==========================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("BEDROCK_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)

# ==========================================
# 2. Helper: Exponential Backoff
# ==========================================

def bedrock_call_with_backoff(model_id: str, body: dict):
    max_attempts = 8
    delay = 1.0

    for attempt in range(max_attempts):
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            return response

        except bedrock.exceptions.ThrottlingException:
            if attempt == max_attempts - 1:
                raise

            sleep_time = delay + random.uniform(0, 0.5)
            print(f"⚠️ Throttled. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            delay *= 2  # exponential backoff

        except Exception as e:
            if "ValidationException" in str(e):
                print(f"❌ Validation Error. Request Body sent: {json.dumps(body, indent=2)}")
            raise RuntimeError(f"Bedrock Error: {e}")

# ==========================================
# 3. LLaMA-3 Specific Formatting
# ==========================================

def format_llama3_prompt(user_message: str, system_message: str) -> str:
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted

def generate_with_bedrock_llama(prompt_content: str, model_id: str):
    system_instruction = (
        "You are a STRICT Data Extraction Engine. "
        "Output ONLY valid XML-like tags. "
        "Do NOT output conversational text."
    )

    final_prompt = format_llama3_prompt(user_message=prompt_content, system_message=system_instruction)

    body = {
        "prompt": final_prompt,
        "max_gen_len": 4096, # Increased for larger questions
        "temperature": 0.1,
        "top_p": 0.9,
    }

    response = bedrock_call_with_backoff(model_id, body)
    data = json.loads(response["body"].read())
    return data["generation"]

# ==========================================
# 4. Data Models (Pydantic)
# ==========================================

class SubSubProblem(BaseModel):
    content: str
    answerable: bool = False
    output_format: str = "text"

class SubProblem(BaseModel):
    content: str
    answerable: bool = False
    output_format: str = "text"
    sub_subproblems: Optional[Dict[str, SubSubProblem]] = Field(default_factory=dict)

class Problem(BaseModel):
    content: str
    subproblems: Optional[Dict[str, SubProblem]] = Field(default_factory=dict)

class ProblemTree(BaseModel):
    problems: Dict[str, Problem]

# ==========================================
# 5. Extraction Logic (Regex)
# ==========================================

def extract_tag_content(text: str, tag_name: str) -> str:
    """Extracts the raw content inside a specific tag."""
    pattern = f"<{tag_name}.*?>(.*?)</{tag_name}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def clean_text_from_tags(raw_text: str, tags_to_remove: list[str]) -> str:
    """
    Removes specific child tag blocks (e.g., <SUBPROBLEM>...</SUBPROBLEM>) 
    from a string to isolate the parent's text content.
    """
    cleaned = raw_text
    for tag in tags_to_remove:
        # Regex to remove <TAG ...> ... </TAG>
        pattern = f"<{tag}.*?>.*?</{tag}>"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def str_to_bool(val: str) -> bool:
    return val.strip().lower() == 'true'

# UPDATED PROMPT: Content tag removed, structure simplified.
TEMPLATE = """
You are a STRICT Data Extraction Engine.

YOUR JOB:
1. Copy the input text EXACTLY.
2. Structure it using ONLY the tags below.
3. Put the text DIRECTLY inside the Problem/Subproblem tags.

REQUIRED STRUCTURE:
---------------------
<PROBLEM id="1">
    ... (Text of the main problem) ...

    <SUBPROBLEM id="(a)">
        ... (Text of subproblem a) ...

        <ANSWERABLE>true/false</ANSWERABLE>
        <OUTPUT_FORMAT>text | image | image+text</OUTPUT_FORMAT>

        <SUBSUB id="i.">
             ... (Text of sub-subproblem i) ...
             <ANSWERABLE>true</ANSWERABLE>
             <OUTPUT_FORMAT>text</OUTPUT_FORMAT>
        </SUBSUB>
    </SUBPROBLEM>
</PROBLEM>
---------------------

RULES:
- Do NOT use <CONTENT> tags. Put text directly in the parent tag.
- Do NOT use <GLOBAL_CONTEXT>.
- Do NOT skip any text from the input.
- Do NOT summarize or fix math/latex.
- If a subproblem exists, the parent <PROBLEM> text should contains ONLY the intro text, not the subproblem text.

INPUT TEXT:
{question_text}

OUTPUT:
"""

def parse_problem_text(raw_text: str) -> Optional[ProblemTree]:
    try:
        print(f"Processing text ({len(raw_text)} chars)...")
        final_user_message = TEMPLATE.replace("{question_text}", raw_text)
        
        LLAMA_MODEL = "meta.llama3-70b-instruct-v1:0"
        
        print(f"Calling {LLAMA_MODEL}...")
        generated_text = generate_with_bedrock_llama(final_user_message, LLAMA_MODEL)
        
        print(f"Raw Output start: {generated_text[:200]}...")

        # 1. Parse Problems
        problem_pattern = r'<PROBLEM\s+id=["\'](.*?)["\']\s*>(.*?)</PROBLEM>'
        problems_found = {}

        for p in re.finditer(problem_pattern, generated_text, re.DOTALL | re.IGNORECASE):
            pid = "Problem " + p.group(1)
            p_body = p.group(2)
            
            # To get the content of PROBLEM, we must strip out all SUBPROBLEM blocks
            p_content = clean_text_from_tags(p_body, ["SUBPROBLEM"])

            # 2. Parse Subproblems inside this Problem
            subproblem_pattern = r'<SUBPROBLEM\s+id=["\'](.*?)["\']\s*>(.*?)</SUBPROBLEM>'
            subprobs_found = {}

            for sp in re.finditer(subproblem_pattern, p_body, re.DOTALL | re.IGNORECASE):
                spid = sp.group(1)
                sp_body = sp.group(2)
                
                # Extract Metadata
                sp_ans_str = extract_tag_content(sp_body, "ANSWERABLE")
                sp_fmt_str = extract_tag_content(sp_body, "OUTPUT_FORMAT")
                
                # To get content of SUBPROBLEM, strip SUBSUB, ANSWERABLE, OUTPUT_FORMAT
                sp_content = clean_text_from_tags(sp_body, ["SUBSUB", "ANSWERABLE", "OUTPUT_FORMAT"])

                # 3. Parse Sub-subproblems
                subsub_pattern = r'<SUBSUB\s+id=["\'](.*?)["\']\s*>(.*?)</SUBSUB>'
                subsub_found = {}

                for ss in re.finditer(subsub_pattern, sp_body, re.DOTALL | re.IGNORECASE):
                    ssid = ss.group(1)
                    ss_body = ss.group(2)
                    
                    ss_ans_str = extract_tag_content(ss_body, "ANSWERABLE")
                    ss_fmt_str = extract_tag_content(ss_body, "OUTPUT_FORMAT")
                    
                    # Content is everything except metadata tags
                    ss_content = clean_text_from_tags(ss_body, ["ANSWERABLE", "OUTPUT_FORMAT"])
                    
                    subsub_found[ssid] = SubSubProblem(
                        content=ss_content,
                        answerable=str_to_bool(ss_ans_str),
                        output_format=ss_fmt_str
                    )

                subprobs_found[spid] = SubProblem(
                    content=sp_content,
                    answerable=str_to_bool(sp_ans_str),
                    output_format=sp_fmt_str,
                    sub_subproblems=subsub_found
                )

            problems_found[pid] = Problem(content=p_content, subproblems=subprobs_found)

        tree = ProblemTree(problems=problems_found)

        if not tree.problems:
            print("❌ No tags detected in output.")
            return None
        else:
            print("✅ Parsed Successfully.")
            return tree

    except Exception as e:
        print(f"❌ Parsing Logic Error: {e}")
        return None

if __name__ == "__main__":
    # Test Data
    sample_text = """
    1. Define X.
       (a) Calculate Y.
       (b) Draw Z.
    """
    result = parse_problem_text(sample_text)
    if result:
        print(result.model_dump_json(indent=2))
