import os
import json
from pathlib import Path
from typing import Set, Optional, List

# --- CONFIGURATION ---
try:
    # Assumes the script is in a 'tools' folder inside the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments where __file__ is not defined
    PROJECT_ROOT = Path.cwd()

OUTPUT_FILENAME = "full_project_source.txt"

# --- EXCLUSION LISTS ---

# Directories to exclude if they appear ANYWHERE in the project structure.
EXCLUDE_DIRS_ANYWHERE: Set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "cache",
    "outputs",
    ".vscode",
    ".idea",
    "venv",
    ".venv",
    "env",
    "build",
    "dist",
    "renv",
    "node_modules",
}

# Directories to exclude ONLY if they are in the project root directory.
# This allows keeping nested directories with the same name (e.g., 'src/app/data').
EXCLUDE_DIRS_ROOT_ONLY: Set[str] = {
    "data",  # User-specific data, not source code
}

# Directory name patterns to exclude (e.g., any directory ending with .egg-info).
EXCLUDE_DIR_PATTERNS: tuple[str, ...] = (".egg-info",)

# File extensions to exclude, typically for binary or non-source files.
EXCLUDE_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".so", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".parquet", ".arrow", ".feather", ".csv", ".zip", ".gz", ".tar", ".rar", ".7z",
    ".db", ".sqlite3",
    ".pdf", ".docx", ".xlsx",
    ".swp", ".swo",
}

# Specific filenames to exclude.
EXCLUDE_FILES: Set[str] = {
    OUTPUT_FILENAME,
    "full_project_source.txt",  # Exclude the old file name just in case
    ".DS_Store",
    "Thumbs.db",
    "celerybeat-schedule",
}


def process_notebook(filepath: Path) -> Optional[str]:
    """
    Parses a Jupyter Notebook (.ipynb) file, extracting only the code and
    markdown content while ignoring all cell outputs.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        content_parts: List[str] = []
        for i, cell in enumerate(notebook.get("cells", [])):
            cell_type = cell.get("cell_type")
            source_list = cell.get("source", [])

            # Ensure 'source' is a single string
            source = "".join(source_list) if isinstance(source_list, list) else str(source_list)

            if not source.strip():
                continue

            if cell_type == "code":
                content_parts.append(f"# --- Code Cell {i+1} ---\n{source}\n")
            elif cell_type == "markdown":
                content_parts.append(f"# --- Markdown Cell {i+1} ---\n{source}\n")

        return "\n".join(content_parts)
    except Exception as e:
        print(f"    [WARN] Could not parse notebook {filepath.name}: {e}")
        return None


def is_likely_text_file(filepath: Path) -> bool:
    """
    Checks if a file is likely to be a text file by checking its extension
    and sniffing the first 1024 bytes for null characters.
    """
    if filepath.suffix.lower() in EXCLUDE_EXTENSIONS:
        return False
    try:
        with open(filepath, "rb") as f:
            # If the first 1KB contains a null byte, it's likely a binary file.
            return b"\0" not in f.read(1024)
    except (IOError, PermissionError):
        return False


def combine_project_files() -> None:
    """
    Scans the project directory, filters out unwanted files/directories,
    and combines all relevant source code into a single text file.
    """
    output_filepath = PROJECT_ROOT / OUTPUT_FILENAME

    print(f"Project root identified as: {PROJECT_ROOT}")
    print(f"Output will be saved to: {output_filepath}\n")

    files_processed_count = 0
    files_skipped_count = 0

    try:
        with open(output_filepath, "w", encoding="utf-8", errors="ignore") as outfile:
            outfile.write("--- Project Source Code Archive ---\n\n")
            outfile.write(
                "This file contains the concatenated source code of the project, "
                "with each file wrapped in tags indicating its relative path.\n\n"
            )

            for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT, topdown=True):
                current_path = Path(dirpath)

                # We filter 'dirnames' in-place to prevent os.walk from recursing into them.
                original_dirs = list(dirnames)  # Make a copy to iterate over
                dirnames.clear()  # Clear the original list to rebuild it

                for d in original_dirs:
                    # Rule 1: Exclude if the directory name should be excluded anywhere.
                    if d in EXCLUDE_DIRS_ANYWHERE:
                        continue
                    # Rule 2: Exclude if it's a root-only-exclusion and we are at the root.
                    if d in EXCLUDE_DIRS_ROOT_ONLY and current_path == PROJECT_ROOT:
                        continue
                    # Rule 3: Exclude if the directory name matches a pattern.
                    if any(d.endswith(p) for p in EXCLUDE_DIR_PATTERNS):
                        continue
                    # If all checks pass, add the directory back to be traversed.
                    dirnames.append(d)

                # --- FILE PROCESSING LOGIC ---
                for filename in sorted(filenames):
                    if filename in EXCLUDE_FILES:
                        continue

                    filepath = current_path / filename
                    relative_path_str = filepath.relative_to(PROJECT_ROOT).as_posix()
                    content: Optional[str] = None

                    try:
                        # Step 1: Specifically handle Jupyter Notebooks.
                        if filepath.suffix.lower() == ".ipynb":
                            print(f"  + Processing Notebook: {relative_path_str}")
                            content = process_notebook(filepath)
                        # Step 2: Handle general text files.
                        elif is_likely_text_file(filepath):
                            print(f"  + Processing Text File: {relative_path_str}")
                            with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
                                content = infile.read()
                        # Step 3: If neither, skip the file.
                        else:
                            print(f"  - Skipping binary/excluded file: {relative_path_str}")
                            files_skipped_count += 1
                            continue

                        # Write content to the output file if it's not empty.
                        if content and content.strip():
                            outfile.write(f"<{relative_path_str}>\n")
                            outfile.write(content.strip())
                            outfile.write(f"\n</{relative_path_str}>\n\n")
                            files_processed_count += 1
                        else:
                            files_skipped_count += 1
                            print(f"    [INFO] No content extracted from {relative_path_str}")

                    except Exception as e:
                        files_skipped_count += 1
                        print(f"    [ERROR] Could not read file {relative_path_str}: {e}")

        print("\n--- Summary ---")
        print(f"Successfully processed {files_processed_count} files.")
        print(f"Skipped {files_skipped_count} binary, excluded, or unreadable files.")
        print(f"Combined output saved to: {output_filepath}")

    except IOError as e:
        print(f"\n[FATAL ERROR] Could not write to output file {output_filepath}: {e}")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    combine_project_files()
