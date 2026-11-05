"""Restore pinned external sources without overwriting unexpected local changes."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'reproduction/vendor_manifest.json').read_text())
for item in manifest:
    destination = root/item['path']
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', item['url'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', item['commit']], check=True)
    current = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if current != item['commit']:
        raise RuntimeError(f'Unexpected revision at {destination}; refusing to change it')
    if item['patch']:
        patch = str(root/item['patch'])
        base = ['git', '-C', str(destination), 'apply']
        if subprocess.run(base+['--reverse', '--check', patch], capture_output=True).returncode == 0:
            print('Already applied:', item['path'])
        else:
            subprocess.run(base+['--check', patch], check=True)
            subprocess.run(base+[patch], check=True)
            print('Applied:', item['path'])
    else:
        print('Pinned:', item['path'])
