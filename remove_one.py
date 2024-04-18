import json
import sys
import re
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process some dataset.")
parser.add_argument('--dataset_name', type=str, default='tky', help='Name of the dataset')

# Parse arguments
args = parser.parse_args()
dataset_name = args.dataset_name

# Load your JSON file
with open(f'/g/data/hn98/peibo/next-poi/dataset/processed/{dataset_name}/train_qa_pairs_kqt.json', 'r') as file:
    data = json.load(file)

# Function to simplify the complex structure
def simplify_structure(text):
    return re.sub(r"\[\{'url': '[^']+', 'name': '([^']+)'}\]", r'\1', text)

# Process each item in the JSON data
for item in data:
    item['question'] = simplify_structure(item['question'])

# Save the modified data back to a JSON file
with open('modified_json_file.json', 'w') as file:
    json.dump(data, file, indent=4)

# Filter out elements where 'question' contains the specified text
filtered_data = [item for item in data if 'is historical data' not in item['question']]

# Save the filtered data back to a JSON file
with open(f'/g/data/hn98/peibo/next-poi/dataset/processed/{dataset_name}/train_qa_pairs_kqt.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)

def substitute_text(line):
    return re.sub(r"\[\{'url': '[^']+', 'name': '([^']+)'}\]", r'\1', line)

# File paths
input_file_path = f'/g/data/hn98/peibo/next-poi/dataset/processed/{dataset_name}/test_qa_pairs_kqt_200.txt'
output_file_path = f'/g/data/hn98/peibo/next-poi/dataset/processed/{dataset_name}/test_qa_pairs_kqt_200.txt'

# Read, process, and write to the new file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        modified_line = substitute_text(line)
        if 'is historical data' not in modified_line:
            output_file.write(modified_line)

