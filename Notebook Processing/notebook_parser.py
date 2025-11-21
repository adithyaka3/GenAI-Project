import sys
import nbformat
from typing import List, Union
from schema import NotebookBlock, CodeBlock, MarkdownBlock, OutputBlock

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
            for output in cell.get('outputs', []):
                out_type = output.output_type
                out_text = None
                out_data = None

                if out_type == 'stream':
                    out_text = output.text
                elif out_type in ('execute_result', 'display_data'):
                    out_data = output.data
                elif out_type == 'error':
                    out_text = f"{output.ename}: {output.evalue}"
                
                processed_outputs.append(OutputBlock(
                    output_type=out_type,
                    text=out_text,
                    data=out_data
                ))

            block = CodeBlock(
                content=cell.source,
                metadata=cell.metadata,
                outputs=processed_outputs
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

        # Optional quick preview (first 3 blocks)
        for b in blocks[:3]:
            print(b)