"""Repository paths (all scripts are run from the code/ directory)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOSS = os.path.join(ROOT, 'results', 'loss_tables')
RESULTS = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'figures')
DATA = os.path.join(ROOT, 'data')
for d in (LOSS, RESULTS, FIGS, DATA):
    os.makedirs(d, exist_ok=True)
