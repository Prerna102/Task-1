from datasets import load_dataset
from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parent.parent

# output_dir = BASE_DIR / "test_pdfs2"
# output_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path("test_pdfs2")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "chainyo/rvl-cdip",
    split="train",
    streaming=True
)

for i, item in enumerate(dataset):
    image = item["image"]

    image.save(output_dir / f"image_{i+1:03d}.png")

    print(f"Saved: {output_dir / f'image_{i+1:03d}.png'}")

    if i == 49:
        break

print("Saved 50 images")