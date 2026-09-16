from pathlib import Path
import time


# Directory containing uploaded files.
UPLOAD_DIR = Path("/home/user/Project/uploads")


# Maximum age of a file: 7 days.
MAX_AGE_SECONDS = 7 * 24 * 60 * 60


# Gets the current time.
now = time.time()


# Checks whether the uploads directory exists.
if not UPLOAD_DIR.exists():
    print("Uploads directory does not exist.")
    raise SystemExit(0)


# Loops through files.
for file_path in UPLOAD_DIR.iterdir():

    # Ignores directories.
    if not file_path.is_file():
        continue

    try:

        # Gets file modification time.
        modified_time = file_path.stat().st_mtime

        # Calculates file age.
        age = now - modified_time

        # Deletes files older than seven days.
        if age > MAX_AGE_SECONDS:

            file_path.unlink()

            print(
                f"Deleted old file: {file_path}"
            )

    except Exception as exc:

        # Logs cleanup failures.
        print(
            f"Could not delete {file_path}: {exc}"
        )

        