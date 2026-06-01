import os
import json
from collections import Counter

def generate_charset():
    base_dir = r"datasets\raw_sources\angelina"
    skip_category = "not_braille"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    label_counter = Counter()

    # Iterate over all categories (directories) in base_dir
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        
        # Skip if it's the not_braille folder or not a directory
        if category == skip_category or not os.path.isdir(category_path):
            continue
            
        print(f"Processing category: {category}...")
        
        # Walk through all JSON files in this category
        for root, _, files in os.walk(category_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Look specifically for the "shapes" array
                        shapes = data.get("shapes", [])
                        
                        # Extract the "label" attribute from each shape
                        for shape in shapes:
                            label = shape.get("label")
                            if label is not None:
                                # Convert to string to ensure JSON serialization works
                                label_counter[str(label)] += 1
                                
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # Write the counter to angelina_charset.json
    output_file = "angelina_charset.json"
    
    # Sort the dictionary by count (highest first) for easier reading
    sorted_labels = dict(sorted(label_counter.items(), key=lambda item: item[1], reverse=True))
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_labels, f, ensure_ascii=False, indent=4)
        print(f"\nSuccessfully wrote charset with {len(sorted_labels)} unique labels to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    generate_charset()
