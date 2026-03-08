import json

with open('notebooks/CICIDS2017/model-training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    # 1. Update intro markdown
    if cell['cell_type'] == 'markdown':
        cell['source'] = [s.replace("Random Forest, Decision Tree, KNN", "Random Forest, Decision Tree") for s in cell['source']]
        
        # 2. Update model comparison markdown
        new_src = []
        for s in cell['source']:
            if "We train three classifiers" in s:
                new_src.append(s.replace("We train three classifiers", "We train two classifiers"))
            elif "3. **K-Nearest Neighbors**" not in s:  # Exclude the 3rd bullet
                new_src.append(s)
        cell['source'] = new_src

    if cell['cell_type'] == 'code':
        # 3. Remove KNN import
        cell['source'] = [s for s in cell['source'] if 'from sklearn.neighbors import KNeighborsClassifier' not in s]
        
        # 4. Update the comparison list
        cell['source'] = [s.replace('all_results = [dt_results, rf_results, knn_results]', 'all_results = [dt_results, rf_results]') for s in cell['source']]

# 5. Remove the KNN markdown and code cells
new_cells = []
for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    if '### 6.3 K-Nearest Neighbors' in src:
        continue
    if 'knn_model = KNeighborsClassifier' in src:
        continue
    new_cells.append(cell)

nb['cells'] = new_cells

with open('notebooks/CICIDS2017/model-training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
    f.write('\n')

