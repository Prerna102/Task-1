from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

DATASET = "lazyc/READoc"
ZIP_FILE = "github.zip"

OUTPUT_DIR = Path("testpdfs2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download the GitHub PDF ZIP
zip_path = hf_hub_download(
    repo_id=DATASET,
    filename=ZIP_FILE,
    repo_type="dataset",
)

print(f"Downloaded: {zip_path}")

# Extract only 50 PDFs
count = 0

with zipfile.ZipFile(zip_path, "r") as z:
    for file_info in z.infolist():

        if not file_info.filename.lower().endswith(".pdf"):
            continue

        # Extract only the PDF itself
        output_file = OUTPUT_DIR / f"pdf_{count + 1:03d}.pdf"

        with z.open(file_info) as source:
            with open(output_file, "wb") as target:
                target.write(source.read())

        count += 1

        print(f"Saved {output_file}")

        if count >= 50:
            break

print(f"\nFinished. Saved {count} PDFs to {OUTPUT_DIR}")