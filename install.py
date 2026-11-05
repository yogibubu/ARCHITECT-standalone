"""Install all bundled MATRIX sources in one dependency-resolution step."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
record = json.loads((root/'release.json').read_text())
for name, expected in record['source_files'].items():
    with (root/name).open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
            raise ValueError('Source checksum mismatch: '+name)
projects = [str(root/'packages'/n) for n in record['components']]
subprocess.run([sys.executable, '-m', 'pip', 'install', *projects, str(root)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
