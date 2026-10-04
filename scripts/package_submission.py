"""Create a portable ZIP containing source, tests, data, evidence and documentation."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "tmp"}


def main():
    target = ROOT.parent / f"{ROOT.name}_submission.zip"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
                   and p.suffix not in {".pyc", ".zip"})
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Submission ZIP verification failed")
    print(f"Packaged {len(files)} files: {target}")


if __name__ == "__main__":
    main()
