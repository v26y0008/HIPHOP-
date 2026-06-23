#!/usr/bin/env python
"""
Run all notebooks in sequence with papermill
"""
import subprocess
import sys
import os

# Notebook execution order
notebooks = [
    'notebooks/02_preprocess.ipynb',
    'notebooks/03_embedding.ipynb',
    'notebooks/04_clustering.ipynb',
    'notebooks/05_network.ipynb',
    'notebooks/06_visualization.ipynb',
]

os.chdir('C:\\Users\\sora2\\hiphop-fan-analysis')

print('🚀 Starting notebook execution pipeline...')
print('=' * 60)

for i, notebook in enumerate(notebooks, 1):
    print(f'\n[{i}/{len(notebooks)}] Executing: {notebook}')
    print('-' * 60)
    
    # Run with jupyter nbconvert
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--output', notebook.replace('.ipynb', '_executed.ipynb'),
        notebook
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f'❌ Error executing {notebook}')
        sys.exit(1)
    else:
        print(f'✅ Successfully executed {notebook}')

print('\n' + '=' * 60)
print('✅ All notebooks executed successfully!')
