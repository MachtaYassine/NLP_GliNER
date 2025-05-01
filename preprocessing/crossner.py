import os
import json
import requests
from tqdm import tqdm
from pilener import save_data_to_file

def download_crossner(domain, output_dir):
    """Download CrossNER dataset for specific domain."""
    domains = ['science', 'politics', 'music', 'literature', 'ai']
    if domain not in domains:
        raise ValueError(f"Domain must be one of {domains}")
    
    base_url = f"https://github.com/zliucr/CrossNER/blob/main/ner_data/{domain}"
    files = ['train.txt', 'dev.txt', 'test.txt']
    
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    for file in files:
        url = f"{base_url}/{file}"
        output_path = os.path.join(domain_dir, file)
        response = requests.get(url)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

def process_crossner_file(filepath):
    """Process CrossNER format into Pile-NER format."""
    processed_data = []
    current_tokens = []
    current_ner = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty or malformed lines
                if not line:
                    if current_tokens:
                        processed_data.append({
                            "tokenized_text": current_tokens,
                            "ner": current_ner
                        })
                        current_tokens = []
                        current_ner = []
                    continue
                
                try:
                    # Handle both tab and space separated formats
                    parts = line.split('\t') if '\t' in line else line.split()
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line {line_num} in {filepath}: {line}")
                        continue
                        
                    token, tag = parts
                    current_tokens.append(token)
                    
                    if tag != 'O':
                        if tag.startswith('B-'):
                            current_ner.append([len(current_tokens)-1, len(current_tokens)-1, tag[2:]])
                        elif tag.startswith('I-') and current_ner:
                            if current_ner[-1][2] == tag[2:]:
                                current_ner[-1][1] = len(current_tokens)-1
                                
                except Exception as e:
                    print(f"Error processing line {line_num} in {filepath}: {str(e)}")
                    continue
            
            # Handle last sentence
            if current_tokens:
                processed_data.append({
                    "tokenized_text": current_tokens,
                    "ner": current_ner
                })
                
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return []

    print(f"Processed {len(processed_data)} sentences from {filepath}")
    return processed_data

def main():
    output_folder = os.path.join(os.path.dirname(__file__), '../data/')
    os.makedirs(output_folder, exist_ok=True)
    
    domains = ['science', 'politics', 'music', 'literature', 'ai']
    
    for domain in tqdm(domains):
        # Download data if file does not exist
        if not os.path.exists(os.path.join(output_folder, domain)):
            print(f"Downloading {domain} dataset...")
            download_crossner(domain, output_folder)
        
        # Process each split
        domain_data = []
        for split in ['train.txt', 'dev.txt', 'test.txt']:
            filepath = os.path.join(output_folder, domain, split)
            domain_data.extend(process_crossner_file(filepath))
        
        # Save processed data
        output_path = os.path.join(output_folder, f'crossner_{domain}_train.json')
        save_data_to_file(domain_data, output_path)
        print(f"{domain} dataset size:", len(domain_data))

if __name__ == "__main__":
    main()