import sys
import nbformat
from typing import List, Union
from schema import NotebookBlock, CodeBlock, MarkdownBlock, OutputBlock
from bedrock_client import invoke_mistral


def summarise_code_block(code: str, outputs_desc: str) -> str:
    """
    Sends the supplied Python code and a description of its outputs to Bedrock (Mistral‑7B‑Instruct)
    and returns a concise, human‑readable summary.

    The prompt includes both the code and any textual or image outputs so the model can
    incorporate that information into the summary.
    """
    prompt = (
        "You are an expert software engineer. Summarise the following Python code "
        "and its execution results in a concise (max 150 words) description. "
        "Mention the purpose, key functions/classes, notable libraries, and what the "
        "outputs (textual results and any images) represent.\n\n"
        f"Code:\n{code}\n\n"
        f"Outputs description:\n{outputs_desc}\n\n"
        "Summary:"
    )
    # Low temperature for deterministic output
    return invoke_mistral(prompt, max_gen_len=512, temperature=0.2, top_p=0.9)


def parse_notebook(file_path: str) -> List[Union[CodeBlock, MarkdownBlock]]:
    """
    Parses a Jupyter notebook file and returns a list of typed blocks.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook {file_path}: {e}")
        return []

    parsed_blocks = []

    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            block = MarkdownBlock(
                content=cell.source,
                metadata=cell.metadata
            )
            parsed_blocks.append(block)

        elif cell.cell_type == 'code':
            # Process outputs
            processed_outputs = []
            outputs_desc_parts = []
            for output in cell.get('outputs', []):
                out_type = output.output_type
                out_text = None
                out_data = None

                if out_type == 'stream':
                    out_text = output.text
                    outputs_desc_parts.append(f"Stream output: {out_text[:200]}{'...' if len(out_text) > 200 else ''}")
                elif out_type in ('execute_result', 'display_data'):
                    out_data = output.data
                    if out_data:
                        if 'image/png' in out_data or 'image/jpeg' in out_data:
                            outputs_desc_parts.append("Image output (e.g., plot, chart)")
                        elif 'text/plain' in out_data:
                            plain_text = out_data['text/plain']
                            outputs_desc_parts.append(f"Text output: {plain_text[:200]}{'...' if len(plain_text) > 200 else ''}")
                elif out_type == 'error':
                    out_text = f"{output.ename}: {output.evalue}"
                    outputs_desc_parts.append(f"Error output: {out_text}")

                processed_outputs.append(OutputBlock(
                    output_type=out_type,
                    text=out_text,
                    data=out_data
                ))

            outputs_desc = "\n".join(outputs_desc_parts) if outputs_desc_parts else "No significant outputs."

            # Generate a summary for the code block before creating the model
            code_summary = summarise_code_block(cell.source, outputs_desc)

            block = CodeBlock(
                content=cell.source,
                metadata=cell.metadata,
                outputs=processed_outputs,
                summary=code_summary
            )
            parsed_blocks.append(block)

    return parsed_blocks


if __name__ == "__main__":
    # Simple test that also saves everything to JSON
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

        # Parse the notebook
        blocks = parse_notebook(notebook_path)
        print(f"Parsed {len(blocks)} blocks.")

        # ----- Save all blocks to a JSON file -----
        import json, os
        # Create a filename like <notebook_name>_parsed.json
        output_path = os.path.splitext(notebook_path)[0] + "_parsed.json"
        with open(output_path, "w", encoding="utf-8") as jf:
            # Pydantic models expose a .dict() method that converts them to plain dicts
            json.dump([b.dict() for b in blocks], jf, ensure_ascii=False, indent=2)
        print(f"Saved full parsed content to {output_path}")

        # ----- Generate high‑level markdown summary -----
        import ast, os
        
        md_lines = []
        md_lines.append(f"# Notebook Summary: {os.path.basename(notebook_path)}\n")
        
        functions = []
        classes = []

        for blk in blocks:
            if isinstance(blk, MarkdownBlock):
                # Add markdown content directly (headings, text, etc.)
                md_lines.append(blk.content)
            elif isinstance(blk, CodeBlock):
                if blk.summary:
                    # Add code summary as a paragraph
                    md_lines.append(f"\n{blk.summary}\n")
                
                # Extract function and class definitions
                try:
                    tree = ast.parse(blk.content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            doc = ast.get_docstring(node) or ""
                            functions.append((node.name, doc))
                        elif isinstance(node, ast.ClassDef):
                            doc = ast.get_docstring(node) or ""
                            classes.append((node.name, doc))
                except Exception:
                    pass

        # Append functions/classes at the end as global context
        if functions:
            md_lines.append("\n## Functions Defined\n")
            for name, doc in functions:
                md_lines.append(f"- **{name}**: {doc if doc else 'No docstring'}")
        if classes:
            md_lines.append("\n## Classes Defined\n")
            for name, doc in classes:
                md_lines.append(f"- **{name}**: {doc if doc else 'No docstring'}")

        md_path = os.path.splitext(notebook_path)[0] + "_summary.md"
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write("\n".join(md_lines))
        print(f"Generated notebook summary markdown at {md_path}")

        # Optional quick preview (first 3 blocks)
        for b in blocks[:3]:
            print(b)