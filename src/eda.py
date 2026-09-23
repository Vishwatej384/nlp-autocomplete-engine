# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

from collections import Counter


def run_eda(sentences, tokenize, top_n=10):
    """Print basic EDA statistics for the corpus."""

    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    all_tokens = [token for tokens in tokenized_sentences for token in tokens]

    word_counts = Counter(all_tokens)

    bigram_counts = Counter()
    trigram_counts = Counter()

    for tokens in tokenized_sentences:
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i + 1])] += 1
        for i in range(len(tokens) - 2):
            trigram_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

    lengths = [len(tokens) for tokens in tokenized_sentences]

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Number of sentences       : {len(sentences)}")
    print(f"Total tokens              : {len(all_tokens)}")
    print(f"Unique tokens             : {len(word_counts)}")
    print(f"Average sentence length   : {sum(lengths) / len(lengths):.2f} words")
    print(f"Minimum sentence length   : {min(lengths)} words")
    print(f"Maximum sentence length   : {max(lengths)} words")

    print("\nTop Words")
    print("-" * 60)
    for word, count in word_counts.most_common(top_n):
        print(f"{word:<25} {count}")

    print("\nTop Bigrams")
    print("-" * 60)
    for (w1, w2), count in bigram_counts.most_common(top_n):
        print(f"{w1} {w2:<22} {count}")

    print("\nTop Trigrams")
    print("-" * 60)
    for (w1, w2, w3), count in trigram_counts.most_common(top_n):
        print(f"{w1} {w2} {w3:<16} {count}")

    print()
