# author : Adithya
import torch
import re
import yaml
import os
from typing import Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# ==========================================
# 1. Configuration & Model Setup
# ==========================================

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

generate_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15,
    return_full_text=False, 
)

# ==========================================
# 2. Pydantic Schema
# ==========================================

class SubSubProblem(BaseModel):
    content: str

class SubProblem(BaseModel):
    content: str
    sub_subproblems: Optional[Dict[str, SubSubProblem]] = Field(default_factory=dict)

class Problem(BaseModel):
    content: str
    subproblems: Optional[Dict[str, SubProblem]] = Field(default_factory=dict)

class ProblemTree(BaseModel):
    global_context: str = ""
    problems: Dict[str, Problem]


# ==========================================
# 3. Prompt
# ==========================================

TEMPLATE = """
<s>[INST]
You are a STRICT Data Extraction Engine.

YOUR GOAL:
Extract the text exactly as it appears. 
Do NOT escape backslashes. Do NOT indent for structure. 
Simply wrap the exact text in the specific tags defined below.

=====================================================
HIERARCHY & TAG RULES
=====================================================

1. **Global Context**: Wrap in `<GLOBAL_CONTEXT> ... </GLOBAL_CONTEXT>`
2. **Problems**: Wrap in `<PROBLEM id="1"> ... </PROBLEM>`. 
   - The `id` should match the question number (e.g., 1, 2).
3. **Content**: Wrap the text description in `<CONTENT> ... </CONTENT>`.
4. **Subproblems**: Wrap in `<SUBPROBLEM id="(a)"> ... </SUBPROBLEM>`.
5. **Sub-subproblems**: Wrap in `<SUBSUB id="i."> ... </SUBSUB>`.

=====================================================
ONE-SHOT EXAMPLE
=====================================================

INPUT:
Let $f(x) = x^2$.
1. Find x.
(a) If x > 0.
i. Verify graph.

OUTPUT:
<GLOBAL_CONTEXT>
Let $f(x) = x^2$.
</GLOBAL_CONTEXT>

<PROBLEM id="1">
  <CONTENT>Find x.</CONTENT>
  <SUBPROBLEM id="(a)">
    <CONTENT>If x > 0.</CONTENT>
    <SUBSUB id="i.">
      <CONTENT>Verify graph.</CONTENT>
    </SUBSUB>
  </SUBPROBLEM>
</PROBLEM>

=====================================================
REAL INPUT TEXT (COPY EXACTLY):
{question_text}

OUTPUT ONLY THE TAGGED TEXT:
[/INST]
"""

# ==========================================
# 4. Robust Regex Parser
# ==========================================

def extract_tag_content(text: str, tag_name: str) -> str:
    """
    Helper to extract content between simple tags like <CONTENT>.
    Case-insensitive matching for tags.
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def parse_problem_text(raw_text: str) -> Optional[ProblemTree]:
    try:
        print(f"Genering Output for {len(raw_text)} chars input...")
        
        # 1. Generate
        final_prompt = TEMPLATE.replace("{question_text}", raw_text)
        outputs = generate_pipe(final_prompt)
        generated_text = outputs[0]["generated_text"]

        # 2. Extract Global Context
        global_context = extract_tag_content(generated_text, "GLOBAL_CONTEXT")
        
        # 3. Extract Problems
        problem_pattern = r'<PROBLEM\s+id\s*=\s*["\'](.*?)["\']\s*>(.*?)</PROBLEM>'
        
        problems_found = {}

        # re.IGNORECASE allows <problem> or <PROBLEM>
        for p_match in re.finditer(problem_pattern, generated_text, re.DOTALL | re.IGNORECASE):
            p_id = "Problem " + p_match.group(1)
            p_body = p_match.group(2)
            p_content = extract_tag_content(p_body, "CONTENT")
            
            # 4. Extract Subproblems
            subproblem_pattern = r'<SUBPROBLEM\s+id\s*=\s*["\'](.*?)["\']\s*>(.*?)</SUBPROBLEM>'
            subprobs_found = {}
            
            for sp_match in re.finditer(subproblem_pattern, p_body, re.DOTALL | re.IGNORECASE):
                sp_id = sp_match.group(1)
                sp_body = sp_match.group(2)
                sp_content = extract_tag_content(sp_body, "CONTENT")
                
                # 5. Extract Sub-Subproblems
                subsub_pattern = r'<SUBSUB\s+id\s*=\s*["\'](.*?)["\']\s*>(.*?)</SUBSUB>'
                subsub_found = {}
                
                for ssp_match in re.finditer(subsub_pattern, sp_body, re.DOTALL | re.IGNORECASE):
                    ssp_id = ssp_match.group(1)
                    ssp_body = ssp_match.group(2)
                    ssp_content = extract_tag_content(ssp_body, "CONTENT")
                    subsub_found[ssp_id] = SubSubProblem(content=ssp_content)

                subprobs_found[sp_id] = SubProblem(content=sp_content, sub_subproblems=subsub_found)

            problems_found[p_id] = Problem(content=p_content, subproblems=subprobs_found)

        tree = ProblemTree(global_context=global_context, problems=problems_found)
        
        # Sanity Check
        if not tree.problems and not tree.global_context:
            print("❌ Parsing returned empty object. Regex failed to match tags.")
            # Uncomment below to debug failures
            # print("RAW TEXT WAS:\n", generated_text)
        else:
            print("✅ Parsed Successfully")
            
        return tree

    except Exception as e:
        print(f"❌ Parsing Error: {e}")
        return None