# author : Debanjan
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import re
import json
import os
import glob
import subprocess
from tqdm.auto import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set environment variable for Nougat
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
input_dir = "input pdfs/"
filename = "DIP_3.pdf"
# Define the PDF path
pdf_path = input_dir + filename

records = []
try:
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            for b in page.get_text("dict")["blocks"]:
                if "lines" not in b: continue
                for l in b["lines"]:
                    for s in l["spans"]:
                        t = s["text"].strip()
                        if not t: continue
                        x0,y0,x1,y1 = s["bbox"]
                        records.append({
                            "page": pno,
                            "text": t,
                            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                            "font": s["font"], "size": s["size"],
                            "bold": "Bold" in s["font"]
                        })
    df = pd.DataFrame(records)
    df["mid_y"] = (df.y0 + df.y1)/2
    print(f"Extracted {len(df)} text spans.")

except Exception as e:
    print(f"Error opening or processing PDF: {e}")
    print("Please make sure 'DIP_3.pdf' is uploaded.")

def group_lines(df, y_gap=5):
    lines=[]
    for page, grp in df.groupby("page"):
        grp=grp.sort_values("y0")
        if grp.empty: continue # Add check for empty group
        buf=[grp.iloc[0]]
        for _,r in list(grp.iterrows())[1:]:
            if r.y0 - buf[-1].y1 < y_gap:
                buf.append(r)
            else:
                if buf: lines.append(buf); buf=[r] # Add check for buf
        if buf: lines.append(buf) # Add check for buf
    out=[]
    for ln in lines:
        if not ln: continue # Skip empty lines
        x0=min(l.x0 for l in ln); y0=min(l.y0 for l in ln)
        x1=max(l.x1 for l in ln); y1=max(l.y1 for l in ln)
        out.append({
            "page":ln[0].page,
            "bbox":[x0,y0,x1,y1],
            "text":" ".join(l.text for l in ln),
            "size":np.mean([l.size for l in ln]),
            "bold":any(l.bold for l in ln)
        })
    return out

# --- THIS FUNCTION IS MODIFIED ---
def detect_sections(lines, gap_thresh=25, size_jump=1.4):
    sections=[]; curr=[]; last=None
    for i,l in enumerate(lines):
        # layout cues
        new=False
        if last is not None:
            if l["page"]!=last["page"]:
                new=True
            else:
                gap = l["bbox"][1]-last["bbox"][3]
                if gap>gap_thresh: new=True
                # *** FIX: Check if l["size"] and last["size"] are non-zero
                elif last["size"] > 0 and l["size"] > 0 and l["size"]/last["size"]>size_jump: new=True
        # heading cues
        if re.match(r"^\d+\.|^[A-Z].*?:", l["text"]) or l["bold"]:
            if last and not new:
                new=True
        # mark-line cue (end of question)
        if last and re.search(r"\bMarks?\b", last["text"], re.I):
            new=True

        if new and curr:
            p=curr[0]["page"]
            x0=min(c["bbox"][0] for c in curr)
            y0=min(c["bbox"][1] for c in curr)
            x1=max(c["bbox"][2] for c in curr)
            y1=max(c["bbox"][3] for c in curr)

            # --- THIS IS THE FIX ---
            # Manually handle the mean for potentially empty lists
            sizes = [c["size"] for c in curr if c["size"] > 0]
            avg_size = np.mean(sizes) if sizes else 0.0
            # --- END OF FIX ---

            sections.append({
                "id": len(sections),
                "page_start":curr[0]["page"],
                "page_end":curr[-1]["page"],
                "bbox":[x0,y0,x1,y1],
                "text":" ".join(c["text"] for c in curr),
                # *** ADDED BOLD AND SIZE METADATA ***
                "bold": any(c["bold"] for c in curr),
                "size": avg_size
            })
            curr=[]
        curr.append(l); last=l

    if curr:
        p=curr[0]["page"]
        x0=min(c["bbox"][0] for c in curr)
        y0=min(c["bbox"][1] for c in curr)
        x1=max(c["bbox"][2] for c in curr)
        y1=max(c["bbox"][3] for c in curr)

        # --- THIS IS THE FIX (applied again for the final block) ---
        sizes = [c["size"] for c in curr if c["size"] > 0]
        avg_size = np.mean(sizes) if sizes else 0.0
        # --- END OF FIX ---

        sections.append({
            "id": len(sections),
            "page_start":curr[0]["page"],
            "page_end":curr[-1]["page"],
            "bbox":[x0,y0,x1,y1],
            "text":" ".join(c["text"] for c in curr),
            # *** ADDED BOLD AND SIZE METADATA ***
            "bold": any(c["bold"] for c in curr),
            "size": avg_size
        })
    return sections

# Run the detection
if 'df' in locals() and not df.empty:
    lines = group_lines(df)
    print(f"{len(lines)} merged text lines found.")
    sections = detect_sections(lines)
    print(f"{len(sections)} logical sections detected.")

    # --- Corrected Block for Visualization (from your code) ---

    # convert all NumPy values to native Python types
    for s in sections:
        s["page_start"] = int(s["page_start"])
        s["page_end"] = int(s["page_end"])
        s["bbox"] = [float(x) for x in s["bbox"]]
        s["text"] = str(s["text"])
        # *** ADDED CONVERSIONS FOR BOLD/SIZE ***
        s["bold"] = bool(s["bold"])
        s["size"] = float(s["size"])

    # --- visualize boxes ---
    pdf_out = "DIP_3_sections.pdf"
    doc_vis = fitz.open(pdf_path)
    for s in sections:
        rect = fitz.Rect(s["bbox"])
        # Only draw on the start page for simplicity
        page = doc_vis[s["page_start"] - 1]
        page.draw_rect(rect, color=(1, 0, 0), width=1.3)
    doc_vis.save(pdf_out)
    doc_vis.close()
    print(f"✅ Section visualization saved → {pdf_out}")
else:
    print("Skipping section detection due to PDF read error or empty DataFrame.")

# --- HELPER FUNCTION (Your original logic only) ---
def cleanup_mmd(mmd_text):
    """
    Cleans the raw .mmd output from Nougat.
    - Converts $...$ to \(...\) and $$...$$ to \[...\]
    (This is the exact logic from your original code)
    """
    if not isinstance(mmd_text, str):
        return ""

    # Fix math delimiters (from your original logic)
    latex = mmd_text.replace("$$", "\n\\[\n")
    latex = latex.replace("$", "\\(")
    latex = latex.replace("\\(\n\\[\n", "\\[")
    latex = latex.replace("\\]\n\\)", "\\]")

    return latex.strip()

# --- MODIFIED PARSING FUNCTION (Vector Cloning) ---
def parse_sections_with_nougat(sections_list, original_pdf_path):

    print(f"Starting Nougat parsing for {len(sections_list)} sections.")
    print("Cloning vector data into temp PDFs. This will be slow but accurate...")

    doc = fitz.open(original_pdf_path)

    # Create temp directories
    temp_pdf_dir = "temp_nougat_pdfs"
    mmd_dir = "nougat_output"
    os.makedirs(temp_pdf_dir, exist_ok=True)
    os.makedirs(mmd_dir, exist_ok=True)

    parsed_sections = []

    for s in tqdm(sections_list):
        temp_pdf_path = None
        mmd_path = None
        try:
            # 1. Get the page and bounding box
            page = doc[s["page_start"] - 1]
            rect = fitz.Rect(s["bbox"])

            # --- THIS IS THE NEW LOGIC ---
            # We only care about the vertical cut (top-bottom).
            # For the horizontal cut (left-right), we'll just use the full page width.
            # This is more robust for single-column text.

            # Get page width
            page_width = page.rect.width

            # Create a new, precise bounding box
            # x0=0, y0=rect.y0, x1=page_width, y1=rect.y1
            # This takes everything from the left to right edge, but only
            # for the vertical slice defined by your section's bbox.
            clip_rect = fitz.Rect(0, rect.y0, page_width, rect.y1)

            # 2. Create a NEW, temporary single-page PDF
            temp_pdf_path = os.path.join(temp_pdf_dir, f"section_{s['id']}.pdf")
            mmd_path = os.path.join(mmd_dir, f"section_{s['id']}.mmd")

            # Create a new blank PDF
            temp_doc = fitz.open()
            # Add a blank page with the same width as the original,
            # but only the height of our clipped section
            temp_page = temp_doc.new_page(width=clip_rect.width, height=clip_rect.height)

            # 3. Clone the vector content from the original page
            # This method copies the actual text, fonts, and paths.
            temp_page.show_pdf_page(
                temp_page.rect,   # Target area (full temp page)
                doc,              # Source document
                page.number,      # Source page number
                clip=clip_rect    # Source area (our clipped section)
            )

            # Save the new 1-page PDF
            temp_doc.save(temp_pdf_path)
            temp_doc.close()

            # 4. Run Nougat on the new high-fidelity, single-page PDF
            subprocess.run(
                [
                    "nougat",
                    temp_pdf_path,
                    "-o", mmd_dir,
                    "--no-skipping"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # 5. Read and CLEAN the Nougat output (.mmd file)
            if os.path.exists(mmd_path):
                with open(mmd_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()

                # Apply *only* your original cleanup function
                latex_content = cleanup_mmd(raw_content)

                s["latex_content"] = latex_content
            else:
                # Fallback to PyMuPDF text if Nougat fails
                s["latex_content"] = cleanup_mmd(s["text"])

            parsed_sections.append(s)

        except Exception as e:
            print(f"Error processing section {s['id']}: {e}")
            s["latex_content"] = cleanup_mmd(s["text"]) # Fallback
            parsed_sections.append(s)

        finally:
            # 6. Clean up temporary files
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            if mmd_path and os.path.exists(mmd_path):
                os.remove(mmd_path)

    doc.close()
    print("✅ Nougat parsing complete.")
    return parsed_sections

# Run the new parsing function
if 'sections' in locals():
    sections_with_latex = parse_sections_with_nougat(sections, pdf_path)
else:
    print("Skipping Nougat parsing as sections were not detected.")

if 'sections_with_latex' in locals():
    # Convert to DataFrame
    df_intermediate = pd.DataFrame(sections_with_latex)

    # Save to CSV
    csv_path = "intermediate_blocks.csv"
    df_intermediate.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"✅ Intermediate results saved to {csv_path}")
    print(df_intermediate.head())
else:
    print("Skipping intermediate save as Nougat parsing did not run.")

import pickle
import pandas as pd
import os

print("=== Preparing data for Kaggle Classification ===")

if 'df_intermediate' not in locals():
    print("Error: 'df_intermediate' DataFrame not found.")
    print("Please run the previous blocks (PyMuPDF, Nougat) to create it.")
else:
    # Ensure dataframe is sorted by ID, which is crucial
    df_to_export = df_intermediate.copy().sort_values(by="id")

    # Prepare export directory
    export_dir = "export_for_kaggle/"
    os.makedirs(export_dir, exist_ok=True)

    # 1) Pickle (faster loading, preserves Python types)
    pkl_path = os.path.join(export_dir, "all_blocks_for_classification.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(df_to_export, f)
    print(f"✅ Saved all {len(df_to_export)} blocks to Pickle: {pkl_path}")

    # 2) CSV version (for easy inspection)
    csv_path = os.path.join(export_dir, "all_blocks_for_classification.csv")
    df_to_export.to_csv(csv_path, index=False)
    print(f"✅ Saved all {len(df_to_export)} blocks to CSV: {csv_path}")

    print("\n=== EXPORT DONE ===")
    print("You can now upload 'all_blocks_for_classification.pkl' to Kaggle.")


