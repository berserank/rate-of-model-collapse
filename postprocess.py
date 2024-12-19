import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def tokenize_text(file_path):
    """Tokenize text from a file into words."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return re.findall(r'\b\w+\b', text)

def find_vocab_size(file_path):
    """Find the unique vocabulary size of a file."""
    words = tokenize_text(file_path)
    vocab = set(words)
    return vocab, len(vocab)

def get_unigram_probability_distribution(file_path):
    """Get unigram probability distribution from a file."""
    words = tokenize_text(file_path)
    unigram_counts = Counter(words)
    total_unigrams = sum(unigram_counts.values())
    return {word: count / total_unigrams for word, count in unigram_counts.items()}

def plot_unigram_histogram(unigram_probabilities, num_words=20):
    """Plot a histogram of unigram probabilities."""
    sorted_unigrams = sorted(unigram_probabilities.items(), key=lambda x: x[1], reverse=True)
    top_unigrams = sorted_unigrams[:num_words]
    words, probabilities = zip(*top_unigrams)

    plt.figure(figsize=(10, 6))
    plt.bar(words, probabilities, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Unigrams')
    plt.ylabel('Probability')
    plt.title(f'Top {num_words} Unigram Probability Distribution')
    plt.tight_layout()
    plt.show()

def analyze_vocab_growth(train_file, generations_folder, output_file):
    """Analyze vocabulary growth across generated files."""
    # Get the vocabulary size of the training file
    train_vocab, _ = find_vocab_size(train_file)

    # Get the list of generated files
    generation_files = sorted(
        [os.path.join(generations_folder, f) for f in os.listdir(generations_folder) if f.startswith("generation") and f.endswith(".txt")],
        key=lambda x: int(re.search(r"generation(\d+)", x).group(1))
    )

    vocab_sizes = [1]
    train_vocab_size = len(train_vocab)
    for file_path in generation_files:
        gen_vocab, _ = find_vocab_size(file_path)
        reduction_ratio = 1-((train_vocab_size - len(gen_vocab)) / train_vocab_size)
        vocab_sizes.append(reduction_ratio)

    vocab_sizes = np.array(vocab_sizes)
    plt.plot(vocab_sizes)
    plt.ylabel('Fraction of Unique Words')
    plt.xlabel('Generation')
    plt.show()
    np.save(output_file, vocab_sizes)

    print(f"Vocabulary size reduction ratios saved to {output_file}")

if __name__ == "__main__":
    train_file = 'train_sampled.txt'
    generations_folder = 'tmp'
    output_file = 'tmp/vocab_sizes.npy'
    analyze_vocab_growth(train_file, generations_folder, output_file)

