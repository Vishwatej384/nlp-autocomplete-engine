# ============================================================
# NLP AUTOCOMPLETE ENGINE
# ============================================================
#
# Features:
# 1. Corpus loading
# 2. Text preprocessing
# 3. Tokenization
# 4. Vocabulary creation
# 5. Bigram language model
# 6. Trigram language model
# 7. Probability calculation
# 8. Laplace smoothing
# 9. Next-word prediction
# 10. Trigram -> Bigram backoff
# 11. Autocomplete
# 12. Top-K suggestions
#
# ============================================================

from evaluation import (
    evaluate_autocomplete,
    display_evaluation
)
from pathlib import Path
import re


# ============================================================
# 1. TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into clean lowercase word tokens.

    Example:

        "Machine Learning is AMAZING!"

    becomes:

        ["machine", "learning", "is", "amazing"]
    """

    # Convert text to lowercase
    text = text.lower()

    # Extract words and simple contractions
    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text
    )

    return tokens


# ============================================================
# 2. LOAD CORPUS
# ============================================================

def load_corpus():
    """
    Load training sentences from data/corpus.txt.
    """

    corpus_path = (
        Path(__file__).parent.parent
        / "data"
        / "corpus.txt"
    )

    # Check whether corpus exists
    if not corpus_path.exists():
        raise FileNotFoundError(
            "Corpus file not found: "
            "data/corpus.txt"
        )

    # Read corpus
    text = corpus_path.read_text(
        encoding="utf-8"
    )

    # Remove empty lines
    sentences = [
        sentence.strip()
        for sentence in text.splitlines()
        if sentence.strip()
    ]

    return sentences


# ============================================================
# 3. BUILD VOCABULARY
# ============================================================

def build_vocabulary(sentences):
    """
    Create a set containing all unique words.
    """

    vocabulary = set()

    for sentence in sentences:

        tokens = tokenize(sentence)

        for token in tokens:

            vocabulary.add(token)

    return vocabulary


# ============================================================
# 4. BUILD N-GRAM MODEL
# ============================================================

def build_ngram_model(sentences):
    """
    Build Bigram and Trigram frequency tables.
    """

    # Bigram:
    #
    # (current_word, next_word)
    #
    bigram_counts = {}

    # Number of times each word
    # appears as a context
    context_counts = {}

    # Trigram:
    #
    # (first_word, second_word, next_word)
    #
    trigram_counts = {}

    # Number of times each two-word
    # context appears
    trigram_context_counts = {}


    # Process every sentence
    for sentence in sentences:

        tokens = tokenize(sentence)


        # ----------------------------------------------------
        # BIGRAMS
        # ----------------------------------------------------

        for i in range(len(tokens) - 1):

            current_word = tokens[i]

            next_word = tokens[i + 1]

            bigram = (
                current_word,
                next_word
            )

            # Count Bigram
            bigram_counts[bigram] = (
                bigram_counts.get(bigram, 0) + 1
            )

            # Count context
            context_counts[current_word] = (
                context_counts.get(current_word, 0) + 1
            )


        # ----------------------------------------------------
        # TRIGRAMS
        # ----------------------------------------------------

        for i in range(len(tokens) - 2):

            first_word = tokens[i]

            second_word = tokens[i + 1]

            next_word = tokens[i + 2]

            trigram = (
                first_word,
                second_word,
                next_word
            )

            # Count Trigram
            trigram_counts[trigram] = (
                trigram_counts.get(trigram, 0) + 1
            )

            # Two-word context
            context = (
                first_word,
                second_word
            )

            trigram_context_counts[context] = (
                trigram_context_counts.get(context, 0) + 1
            )


    return (
        bigram_counts,
        context_counts,
        trigram_counts,
        trigram_context_counts
    )


# ============================================================
# 5. BIGRAM PROBABILITY
# ============================================================

def bigram_probability(
    current_word,
    next_word,
    bigram_counts,
    context_counts,
    vocabulary_size
):
    """
    Calculate Laplace-smoothed Bigram probability.

    Formula:

        P(next | current)
        =
        (Count(current,next) + 1)
        /
        (Count(current) + V)
    """

    current_word = current_word.lower()

    next_word = next_word.lower()

    bigram = (
        current_word,
        next_word
    )

    # Get Bigram count
    count = bigram_counts.get(
        bigram,
        0
    )

    # Get context count
    context_count = context_counts.get(
        current_word,
        0
    )

    # Laplace smoothing
    probability = (
        (count + 1)
        /
        (context_count + vocabulary_size)
    )

    return probability


# ============================================================
# 6. TRIGRAM PROBABILITY
# ============================================================

def trigram_probability(
    first_word,
    second_word,
    next_word,
    trigram_counts,
    trigram_context_counts,
    vocabulary_size
):
    """
    Calculate Laplace-smoothed Trigram probability.

    Formula:

        P(next | first, second)
        =
        (Count(first,second,next) + 1)
        /
        (Count(first,second) + V)
    """

    first_word = first_word.lower()

    second_word = second_word.lower()

    next_word = next_word.lower()

    trigram = (
        first_word,
        second_word,
        next_word
    )

    context = (
        first_word,
        second_word
    )

    # Get Trigram count
    count = trigram_counts.get(
        trigram,
        0
    )

    # Get two-word context count
    context_count = trigram_context_counts.get(
        context,
        0
    )

    # Laplace smoothing
    probability = (
        (count + 1)
        /
        (context_count + vocabulary_size)
    )

    return probability


# ============================================================
# 7. BIGRAM PREDICTION
# ============================================================

def predict_bigram(
    current_word,
    bigram_counts,
    context_counts,
    vocabulary,
    top_n=3,
    use_smoothing=False
):
    """
    Predict the next word using a Bigram model.
    """

    current_word = current_word.lower()

    predictions = []

    vocabulary_size = len(vocabulary)


    # Check every possible next word
    for next_word in vocabulary:

        if use_smoothing:

            probability = bigram_probability(
                current_word,
                next_word,
                bigram_counts,
                context_counts,
                vocabulary_size
            )

        else:

            bigram = (
                current_word,
                next_word
            )

            count = bigram_counts.get(
                bigram,
                0
            )

            context_count = context_counts.get(
                current_word,
                0
            )

            # Skip if context doesn't exist
            if context_count == 0:
                continue

            probability = (
                count / context_count
            )

            # Only return words that
            # actually appeared
            if count == 0:
                continue


        predictions.append(
            (
                next_word,
                probability
            )
        )


    # Sort highest probability first
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions[:top_n]


# ============================================================
# 8. TRIGRAM PREDICTION
# ============================================================

def predict_trigram(
    first_word,
    second_word,
    trigram_counts,
    trigram_context_counts,
    vocabulary,
    top_n=3,
    use_smoothing=False
):
    """
    Predict the next word using a Trigram model.
    """

    first_word = first_word.lower()

    second_word = second_word.lower()

    predictions = []

    vocabulary_size = len(vocabulary)

    context = (
        first_word,
        second_word
    )


    # Check every possible next word
    for next_word in vocabulary:

        if use_smoothing:

            probability = trigram_probability(
                first_word,
                second_word,
                next_word,
                trigram_counts,
                trigram_context_counts,
                vocabulary_size
            )

        else:

            trigram = (
                first_word,
                second_word,
                next_word
            )

            count = trigram_counts.get(
                trigram,
                0
            )

            context_count = (
                trigram_context_counts.get(
                    context,
                    0
                )
            )

            # Skip if context doesn't exist
            if context_count == 0:
                continue

            probability = (
                count / context_count
            )

            # Only return observed words
            if count == 0:
                continue


        predictions.append(
            (
                next_word,
                probability
            )
        )


    # Sort highest probability first
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return predictions[:top_n]


# ============================================================
# 9. AUTOCOMPLETE
# ============================================================

def autocomplete(
    text,
    bigram_counts,
    context_counts,
    trigram_counts,
    trigram_context_counts,
    vocabulary,
    top_n=3,
    use_smoothing=False
):
    """
    Generate next-word suggestions.

    Strategy:

        2 or more words
            ↓
        Try Trigram
            ↓
        If unavailable
            ↓
        Fall back to Bigram
    """

    # Tokenize input
    tokens = tokenize(text)


    # No input
    if len(tokens) == 0:

        return []


    # --------------------------------------------------------
    # TRY TRIGRAM
    # --------------------------------------------------------

    if len(tokens) >= 2:

        first_word = tokens[-2]

        second_word = tokens[-1]

        predictions = predict_trigram(
            first_word,
            second_word,
            trigram_counts,
            trigram_context_counts,
            vocabulary,
            top_n,
            use_smoothing
        )

        # If predictions exist,
        # use Trigram model
        if predictions:

            return predictions


    # --------------------------------------------------------
    # FALL BACK TO BIGRAM
    # --------------------------------------------------------

    last_word = tokens[-1]

    predictions = predict_bigram(
        last_word,
        bigram_counts,
        context_counts,
        vocabulary,
        top_n,
        use_smoothing
    )

    return predictions


# ============================================================
# 10. DISPLAY PREDICTIONS
# ============================================================

def display_predictions(predictions):

    if not predictions:

        print("No predictions found.")

        return


    for word, probability in predictions:

        print(
            f"{word:<20} "
            f"{probability:.2%}"
        )


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print("NLP AUTOCOMPLETE ENGINE")

    print("=" * 60)

    print()


    # --------------------------------------------------------
    # LOAD CORPUS
    # --------------------------------------------------------

    sentences = load_corpus()
    split_index = int(len(sentences) * 0.8)
    train_sentences = sentences[:split_index]
    test_sentences = sentences[split_index:]


    print(
        "Training sentences:",
        len(train_sentences)
    )

    print(
        "Test sentences:",
        len(test_sentences)
    )

    print(
        "Corpus loaded successfully!"
    )

    print(
        "Number of sentences:",
        len(sentences)
    )

    print()


    # --------------------------------------------------------
    # BUILD VOCABULARY
    # --------------------------------------------------------
    vocabulary = build_vocabulary(train_sentences)

    print(
        "Vocabulary size:",
        len(vocabulary)
    )

    print()


    # --------------------------------------------------------
    # BUILD N-GRAM MODEL
    # --------------------------------------------------------

    (
        bigram_counts,
        context_counts,
        trigram_counts,
        trigram_context_counts
    ) = build_ngram_model(train_sentences)


    print(
        "Number of Bigrams:",
        len(bigram_counts)
    )

    print(
        "Number of Trigrams:",
        len(trigram_counts)
    )

    print()


    # --------------------------------------------------------
    # TOKENIZER TEST
    # --------------------------------------------------------

    print("Tokenizer Test")

    print("-" * 30)

    test_text = (
        "Machine Learning is AMAZING!"
    )

    print(
        "Original:",
        test_text
    )

    print(
        "Tokens:",
        tokenize(test_text)
    )

    print()


    # --------------------------------------------------------
    # BIGRAM TEST
    # --------------------------------------------------------

    print("Bigram Prediction")

    print("-" * 30)

    predictions = predict_bigram(
        "machine",
        bigram_counts,
        context_counts,
        vocabulary,
        top_n=3
    )

    display_predictions(
        predictions
    )

    print()


    # --------------------------------------------------------
    # TRIGRAM TEST
    # --------------------------------------------------------

    print("Trigram Prediction")

    print("-" * 30)

    predictions = predict_trigram(
        "machine",
        "learning",
        trigram_counts,
        trigram_context_counts,
        vocabulary,
        top_n=3
    )

    display_predictions(
        predictions
    )

    print()


    # --------------------------------------------------------
    # AUTOCOMPLETE TEST
    # --------------------------------------------------------

    print("Autocomplete")

    print("-" * 30)

    test_input = "machine learning"

    print(
        "Input:",
        test_input
    )

    print()

    print("Suggestions:")

    predictions = autocomplete(
        test_input,
        bigram_counts,
        context_counts,
        trigram_counts,
        trigram_context_counts,
        vocabulary,
        top_n=3
    )

    display_predictions(
        predictions
    )

    print()


    # --------------------------------------------------------
    # SMOOTHING TEST
    # --------------------------------------------------------

    print("Laplace Smoothing")

    print("-" * 30)

    probability = bigram_probability(
        "machine",
        "learning",
        bigram_counts,
        context_counts,
        len(vocabulary)
    )

    print(
        "P(learning | machine) =",
        f"{probability:.6f}"
    )


    # Test an unseen Bigram
    probability = bigram_probability(
        "machine",
        "unknownword",
        bigram_counts,
        context_counts,
        len(vocabulary)
    )

    print(
        "P(unknownword | machine) =",
        f"{probability:.6f}"
    )

    print()


    # --------------------------------------------------------
    # INTERACTIVE MODE
    # --------------------------------------------------------

    print("=" * 60)

    print("INTERACTIVE AUTOCOMPLETE")

    print("=" * 60)

    print(
        "Type a sentence to get suggestions."
    )

    print(
        "Type 'exit' to stop."
    )

    print()


    while True:

        user_input = input(
            "Enter text: "
        )

        # Exit
        if user_input.lower().strip() == "exit":

            print(
                "Autocomplete engine stopped."
            )

            break


        predictions = autocomplete(
            user_input,
            bigram_counts,
            context_counts,
            trigram_counts,
            trigram_context_counts,
            vocabulary,
            top_n=5
        )
        results = evaluate_autocomplete(
            test_sentences,
            bigram_counts,
            context_counts,
            trigram_counts,
            trigram_context_counts,
            vocabulary
        )
        display_evaluation(results)

        print()

        print("Suggestions:")

        display_predictions(
            predictions
        )

        print()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()