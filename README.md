# GenAI Project

- Authors of each code has been written at the top of the file. In case of any clarification, please contact them.

## Execution Order:
1. create a python environment and run
``` 
pip install -r requirements.txt
```
2. Run 
```
preclassification.py
```
Outputs from this Python file will be stored in the *export_for_kaggle* folder. The only important components from this is the .pkl file

3. Run 
```
classifyAllBlocks.py
```
Outputs from this file will be stored in the *classifiedBlocksOutput* directory. The file *qwen_debug_full_outputs.csv* contains the tagging of each text block as "start of a new question" or "continuation block". 

4. Run
```
extractQuestionFromCsv.py
```
This file imports a function from the subpart_split.py file which is the function call to the model to split into subparts. For each question this function call will be made and for each question a corresponding .yaml file will be stored in the *yaml_parsed_questions* directory.

Model being used in subpart_split.py is Mistral-8B which is being quantized as of now. If AWS integration to backend is done, quantization can be removed.


### Additional Things to be implemented
- Partition of answers notebook 
- Similarity correlation between answers and question code blocks
- Frontend/Backend for tool
- AWS backend to be able to test large models without quantization

