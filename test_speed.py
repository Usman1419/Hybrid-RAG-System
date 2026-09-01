import time
import pymupdf
import sys
from pathlib import Path

pdf_path = Path("data/pdfs/pdf1.pdf")

if not pdf_path.exists():
    print(f"Error: Could not find {pdf_path}")
    print("Please place a PDF file there or update the path to test.")
    sys.exit(1)

doc = pymupdf.open(pdf_path)

start = time.time()
pages_to_test = min(10, len(doc))

for i in range(pages_to_test):
    text = doc[i].get_text()

elapsed = time.time() - start

print(f"{pages_to_test} pages took: {elapsed:.2f} seconds")

if pages_to_test > 0:
    estimated_minutes = (elapsed / pages_to_test) * len(doc) / 60
    print(f"Estimated full PDF ({len(doc)} pages): {estimated_minutes:.2f} minutes")

doc.close()