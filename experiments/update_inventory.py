"""Refresh the inventory of authored experiments and their local run outputs."""
import csv
from pathlib import Path


def main():
    repository = Path(__file__).resolve().parents[1]
    workspace = repository.parent
    destination = repository / "experiments" / "FILE_INVENTORY.csv"
    entries = {}
    for path in (repository / "experiments").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            entries[path.relative_to(workspace).as_posix()] = "experiment_source_or_documentation"
    prefixes = ("marco-smoke", "validation-training-cache", "validation-15-5-",
                "feature-analysis-15-", "expanded-training-15-", "expanded-stability-15-",
                "selection-validation-15-5-", "seed-robustness-15-5-")
    for directory in (workspace / "runs").iterdir():
        if not directory.is_dir() or not directory.name.startswith(prefixes):
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            category = "copied_smoke_input" if "inputs" in path.parts else "generated_run_output"
            if path.parent.name == "marco-smoke" and path.name in ("run_smoke.py", "README.md"):
                category = "smoke_support"
            entries[path.relative_to(workspace).as_posix()] = category
    entries[destination.relative_to(workspace).as_posix()] = "generated_inventory"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "category"])
        writer.writerows(sorted(entries.items()))
    print(f"Recorded {len(entries)} file paths in {destination}")


if __name__ == "__main__":
    main()
