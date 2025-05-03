import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_json(filepath):
    """Loads a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_nb_of_tokens(data):
    """Analyzes the number of tokens in the dataset."""
    return sum(len(entry['tokenized_text']) for entry in data)

def analyze_sentence_sizes(data):
    """Analyzes the size of sentences in the dataset."""
    return [len(entry['tokenized_text']) for entry in data]

def analyze_entity_distribution(data):
    """Analyzes the distribution of entities in the dataset."""
    entity_counter = Counter()
    for entry in data:
        for _, _, entity_type in entry['ner']:
            entity_counter[entity_type] += 1
    return entity_counter

def analyze_entity_overlap(data1, data2):
    """Analyzes the overlap of entities between two datasets."""
    entities1 = set((tuple(entry['tokenized_text'][span[0]:span[1]+1]), span[2]) for entry in data1 for span in entry['ner'])
    entities2 = set((tuple(entry['tokenized_text'][span[0]:span[1]+1]), span[2]) for entry in data2 for span in entry['ner'])
    return len(entities1 & entities2), len(entities1), len(entities2)

def plot_sentence_length_distribution(datasets, output_folder):
    """Plots the distribution of sentence lengths for each dataset."""
    dataset_names = {
        'crossner_ai_train.json': 'AI',
        'crossner_literature_train.json': 'Literature',
        'crossner_music_train.json': 'Music',
        'crossner_politics_train.json': 'Politics',
        'crossner_science_train.json': 'Science',
        'pilener_train.json': 'PileNER'
    }
    
    plt.figure(figsize=(10, 6))
    for name, data in datasets.items():
        sentence_lengths = analyze_sentence_sizes(data)
        sns.kdeplot(sentence_lengths, label=dataset_names[name], fill=True, alpha=0.5)
    plt.title("Sentence Length Distribution")
    plt.xlabel("Sentence Length")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "sentence_length_distribution.png"))
    plt.close()

def plot_entity_overlap_heatmap_with_pilener(datasets, output_folder):
    """Plots a heatmap of the entity overlap ratio between CrossNER datasets and PileNER."""
    dataset_names = {
        'crossner_ai_train.json': 'AI',
        'crossner_literature_train.json': 'Literature',
        'crossner_music_train.json': 'Music',
        'crossner_politics_train.json': 'Politics',
        'crossner_science_train.json': 'Science',
        'pilener_train.json': 'PileNER'
    }
    
    crossner_names = [name for name in datasets if name != 'pilener_train.json']
    pilener_data = datasets['pilener_train.json']
    overlap_ratios = []

    for name in crossner_names:
        overlap, crossner_size, _ = analyze_entity_overlap(datasets[name], pilener_data)
        ratio = (overlap / crossner_size) * 100 if crossner_size > 0 else 0
        overlap_ratios.append(ratio)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.array(overlap_ratios).reshape(-1, 1),
        annot=True,
        fmt=".2f",  # Display as a percentage with two decimal places
        xticklabels=['PileNER'],
        yticklabels=[dataset_names[name] for name in crossner_names],
        cmap="Blues"
    )
    plt.title("Entity Overlap Heatmap in Percent (CrossNER vs PileNER)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "entity_overlap_heatmap.png"))
    plt.close()

def compare_datasets():
    """Compares datasets on various analytics and generates plots."""
    base_folder = os.path.join(os.path.dirname(__file__), '../data/')
    output_folder = os.path.join(os.path.dirname(__file__), '../images/')
    os.makedirs(output_folder, exist_ok=True)

    files = [
        'crossner_ai_train.json',
        'crossner_literature_train.json',
        'crossner_music_train.json',
        'crossner_politics_train.json',
        'crossner_science_train.json',
        'pilener_train.json'
    ]
    
    datasets = {file: load_json(os.path.join(base_folder, file)) for file in files}
    
    # Analytics
    for name, data in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Number of tokens: {analyze_nb_of_tokens(data)}")
        print(f"  Sentence sizes: {analyze_sentence_sizes(data)[:10]} (first 10)")
        print(f"  Entity distribution: {analyze_entity_distribution(data)}")
        print()
    
    # Entity overlap
    pilener_data = datasets['pilener_train.json']
    for name, data in datasets.items():
        if name != 'pilener_train.json':
            overlap, size1, size2 = analyze_entity_overlap(pilener_data, data)
            print(f"Entity overlap between pilener_train.json and {name}:")
            print(f"  Overlap: {overlap}")
            print(f"  Pilener entities: {size1}")
            print(f"  {name} entities: {size2}")
            print()
    
    # Generate plots
    plot_sentence_length_distribution(datasets, output_folder)
    plot_entity_overlap_heatmap_with_pilener(datasets, output_folder)

if __name__ == "__main__":
    compare_datasets()
