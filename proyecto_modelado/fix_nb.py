import json
filepath = 'notebooks/03_model_evaluation.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    d = json.load(f)

# The cell is at index 3, and the source line is at index 3
# Let's search for the line in all cells to be safe
for cell in d.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'print(f"\\n{\\"=\\"*40}\\n")' in line:
                source[i] = line.replace('print(f"\\n{\\"=\\"*40}\\n")', 'print("\\n" + "="*40 + "\\n")')

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
print("Notebook fixed")
