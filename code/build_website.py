from pathlib import Path
import shutil
root = Path(__file__).resolve().parents[1]
for path in (root / 'website').rglob('*'):
    if path.is_file():
        target = root / 'docs' / path.relative_to(root / 'website')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
print('Built docs/ from website/')
