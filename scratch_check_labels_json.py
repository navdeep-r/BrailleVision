import os
import json
import collections

cnt = collections.Counter()
for root, _, files in os.walk('datasets/raw_sources/angelina'):
    for fname in files:
        if fname.endswith('.json'):
            path = os.path.join(root, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                marks = data.get("marks", data.get("labeled_cells", data.get("cells", data.get("shapes", []))))
                for mark in marks:
                    if "label" in mark:
                        char = str(mark["label"]).lower()
                    elif "char" in mark:
                        char = str(mark["char"]).lower()
                    elif "character" in mark:
                        char = str(mark["character"]).lower()
                    else:
                        char = "?"
                    cnt.update([char])
            except:
                pass

with open('scratch_counts.json', 'w', encoding='utf-8') as f:
    json.dump(cnt.most_common(100), f, ensure_ascii=False, indent=2)
