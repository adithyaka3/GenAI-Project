# author : Adithya
import pandas as pd
from subpart_split import parse_problem_text
import yaml
import os

def save_problem_to_file(problem_obj, filename: str):
    if problem_obj is None:
        print(f"⚠️ Skipping {filename} — Result is None.")
        return

    # Check if the object is empty
    if not problem_obj.problems:
        print(f"⚠️ Warning: {filename} was saved but contains NO data (Regex failed).")

    # model_dump(mode='json') automatically handles the nested structure
    data = problem_obj.model_dump(mode='json')

    with open(filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data, 
            f, 
            sort_keys=False, 
            allow_unicode=True, 
            default_flow_style=False,
            width=1000
        )
    print(f"✅ Saved: {filename}")


def extract_questions_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Normalize columns
    df['latex_content'] = df['latex_content'].fillna('')
    df['question_start_type'] = df['question_start_type'].fillna('')
    
    questions_list = []
    current_question_parts = []
    
    for index, row in df.iterrows():
        # normalize to lowercase for safety
        start_type = str(row['question_start_type']).strip().lower() 
        text_segment = str(row['latex_content'])
        
        # Check for variation in string formatting
        if 'start of a new question' in start_type:
            if current_question_parts:
                questions_list.append("\n".join(current_question_parts))
                current_question_parts = []
            
            current_question_parts.append(text_segment)
        else:
            if current_question_parts or (not current_question_parts and text_segment.strip()):
                current_question_parts.append(text_segment)
    
    if current_question_parts:
        questions_list.append("\n".join(current_question_parts))
        
    return questions_list

# -------------------------
# RUNNING THE EXTRACTION
# -------------------------

if __name__ == "__main__":
    file_name = 'classifiedBlocksOutput/qwen_debug_full_outputs.csv'
    
    if not os.path.exists(file_name):
        print(f"❌ Error: {file_name} not found.")
        exit()

    questions = extract_questions_from_csv(file_name)
    print(f"Successfully extracted {len(questions)} questions.")

    output_dir = 'yaml_parsed_questions'
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous runs
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    # Process every question
    for i, question in enumerate(questions):
        print("="*40)
        print(f"\nProcessing question {i+1}/{len(questions)}...")
        
        if not question.strip():
            print("Skipping empty text block.")
            continue

        parsed = parse_problem_text(question)
        save_problem_to_file(parsed, f'{output_dir}/parsed_question_{i+1}.yaml')

    print("\nAll done.")
