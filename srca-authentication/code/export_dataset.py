"""Write the 760-patch evaluation set to data/patches/ as PNG files with a manifest (SHA-256).
The patches are generated deterministically (seed 0) from images bundled with scikit-image,
exactly as used by precompute.py. Run:  python export_dataset.py   (add --verify to check hashes)."""
import os, sys, csv, hashlib, numpy as np
from PIL import Image
from paths import DATA
from data_attacks import load_pool

pool = load_pool({'natural': 40, 'biomedical': 60}, seed=0)
man = os.path.join(DATA, 'manifest.csv')
if '--verify' in sys.argv:
    ok = 0
    for row in csv.DictReader(open(man)):
        h = hashlib.sha256(pool[int(row['id'])]['img'].tobytes()).hexdigest()
        ok += h == row['sha256_pixels']
    print(f'{ok}/{len(pool)} patches match the manifest'); sys.exit(0)
rows = []
for i, P in enumerate(pool):
    rel = os.path.join('patches', P['domain'], f"{i:04d}_{P['source']}.png")
    path = os.path.join(DATA, rel); os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(P['img']).save(path, optimize=True)
    rows.append(dict(id=i, domain=P['domain'], source=P['source'], file=rel,
                     sha256_pixels=hashlib.sha256(P['img'].tobytes()).hexdigest()))
with open(man, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} patches and {man}')
