from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile
import pymupdf
import io

DATASET = "lazyc/READoc"
ZIP_FILE = "github.zip"

OUTPUT_DIR = Path("testpdfs2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PAGES = 1
MAX_PAGES = 4
TARGET_PDFS = 50

# Download the ZIP
zip_path = hf_hub_download(
    repo_id=DATASET,
    filename=ZIP_FILE,
    repo_type="dataset",
)

print(f"Downloaded ZIP: {zip_path}")

count = 0
checked = 0

with zipfile.ZipFile(zip_path, "r") as z:

    for file_info in z.infolist():

        if not file_info.filename.lower().endswith(".pdf"):
            continue

        checked += 1

        try:
            # Read PDF directly from ZIP into memory
            pdf_bytes = z.read(file_info)

            # Open PDF from memory
            pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            page_count = len(pdf)

            print(
                f"Checking {file_info.filename} "
                f"-> {page_count} pages"
            )

            # Only keep PDFs with 1-4 pages
            if MIN_PAGES <= page_count <= MAX_PAGES:

                output_file = OUTPUT_DIR / f"pdf_{count + 1:03d}.pdf"

                with open(output_file, "wb") as target:
                    target.write(pdf_bytes)

                count += 1

                print(
                    f"  SAVED: {output_file} "
                    f"({page_count} pages)"
                )

            pdf.close()

        except Exception as e:
            print(f"  ERROR: {file_info.filename}: {e}")

        # Stop after getting 50 suitable PDFs
        if count >= TARGET_PDFS:
            break

print("\nFinished.")
print(f"PDFs checked: {checked}")
print(f"PDFs saved:   {count}")
print(f"Output folder: {OUTPUT_DIR.resolve()}")
