# ============================================================
# MODEL EVALUATION AND COMPARISON
# ============================================================

import math


def _top_k(predictions, k):
    return [word for word, _ in predictions[:k]]


def _predict_bigram_for_eval(word, bigram_counts, context_counts, vocabulary):
    context_count = context_counts.get(word, 0)
    if context_count == 0:
        return []

    predictions = []
    for candidate in vocabulary:
        count = bigram_counts.get((word, candidate), 0)
        if count > 0:
            predictions.append((candidate, count / context_count))

    predictions.sort(key=lambda item: (-item[1], item[0]))
    return predictions


def _predict_trigram_for_eval(first, second, trigram_counts,
                              trigram_context_counts, vocabulary):
    context = (first, second)
    context_count = trigram_context_counts.get(context, 0)
    if context_count == 0:
        return []

    predictions = []
    for candidate in vocabulary:
        count = trigram_counts.get((first, second, candidate), 0)
        if count > 0:
            predictions.append((candidate, count / context_count))

    predictions.sort(key=lambda item: (-item[1], item[0]))
    return predictions


def _laplace_bigram_probability(word, next_word, bigram_counts,
                                context_counts, vocabulary_size):
    count = bigram_counts.get((word, next_word), 0)
    context_count = context_counts.get(word, 0)
    return (count + 1) / (context_count + vocabulary_size)


def _laplace_trigram_probability(first, second, next_word,
                                 trigram_counts, trigram_context_counts,
                                 vocabulary_size):
    count = trigram_counts.get((first, second, next_word), 0)
    context_count = trigram_context_counts.get((first, second), 0)
    return (count + 1) / (context_count + vocabulary_size)


def _accuracy_for_predictions(prediction_lists, targets):
    total = len(targets)
    if total == 0:
        return {1: 0.0, 3: 0.0, 5: 0.0, "total": 0}

    result = {1: 0, 3: 0, 5: 0}
    for predictions, target in zip(prediction_lists, targets):
        for k in (1, 3, 5):
            if target in _top_k(predictions, k):
                result[k] += 1

    return {
        1: 100 * result[1] / total,
        3: 100 * result[3] / total,
        5: 100 * result[5] / total,
        "total": total,
    }


def _test_events(test_sentences, tokenize):
    """Return (tokens, index) pairs for every next-word prediction event."""
    events = []
    for sentence in test_sentences:
        tokens = tokenize(sentence)
        for i in range(1, len(tokens)):
            events.append((tokens, i))
    return events


def compare_models(test_sentences, bigram_counts, context_counts,
                   trigram_counts, trigram_context_counts, vocabulary,
                   tokenize, predict_bigram, predict_trigram):
    """Compare Bigram, Trigram, and Trigram->Bigram Backoff models."""

    events = _test_events(test_sentences, tokenize)
    vocab_size = len(vocabulary)

    model_predictions = {
        "Bigram": [],
        "Trigram": [],
        "Trigram + Bigram Backoff": [],
    }
    targets = {
        "Bigram": [],
        "Trigram": [],
        "Trigram + Bigram Backoff": [],
    }

    # Perplexity log-probabilities. Trigram events start at i=2.
    log_probs = {name: [] for name in model_predictions}

    for tokens, i in events:
        target = tokens[i]

        # Bigram model
        bigram_preds = predict_bigram(
            tokens[i - 1], bigram_counts, context_counts,
            vocabulary, top_n=5, use_smoothing=False
        )
        model_predictions["Bigram"].append(bigram_preds)
        targets["Bigram"].append(target)
        log_probs["Bigram"].append(math.log(
            _laplace_bigram_probability(
                tokens[i - 1], target, bigram_counts,
                context_counts, vocab_size
            )
        ))

        # Trigram model: when there is no two-word context, no prediction
        # is available for this event. It is excluded from accuracy totals.
        if i >= 2:
            trigram_preds = predict_trigram(
                tokens[i - 2], tokens[i - 1], trigram_counts,
                trigram_context_counts, vocabulary,
                top_n=5, use_smoothing=False
            )
            if trigram_preds:
                model_predictions["Trigram"].append(trigram_preds)
                targets["Trigram"].append(target)
                log_probs["Trigram"].append(math.log(
                    _laplace_trigram_probability(
                        tokens[i - 2], tokens[i - 1], target,
                        trigram_counts, trigram_context_counts,
                        vocab_size
                    )
                ))

        # Actual autocomplete strategy: Trigram first, Bigram fallback.
        if i >= 2:
            backoff_preds = predict_trigram(
                tokens[i - 2], tokens[i - 1], trigram_counts,
                trigram_context_counts, vocabulary,
                top_n=5, use_smoothing=False
            )
        else:
            backoff_preds = []

        if backoff_preds:
            model_predictions["Trigram + Bigram Backoff"].append(backoff_preds)
            targets["Trigram + Bigram Backoff"].append(target)
            log_probs["Trigram + Bigram Backoff"].append(math.log(
                _laplace_trigram_probability(
                    tokens[i - 2], tokens[i - 1], target,
                    trigram_counts, trigram_context_counts,
                    vocab_size
                )
            ))
        else:
            model_predictions["Trigram + Bigram Backoff"].append(bigram_preds)
            targets["Trigram + Bigram Backoff"].append(target)
            log_probs["Trigram + Bigram Backoff"].append(math.log(
                _laplace_bigram_probability(
                    tokens[i - 1], target, bigram_counts,
                    context_counts, vocab_size
                )
            ))

    results = {}
    for name in model_predictions:
        scores = _accuracy_for_predictions(
            model_predictions[name], targets[name]
        )
        perplexity = math.exp(
            -sum(log_probs[name]) / len(log_probs[name])
        ) if log_probs[name] else float("inf")

        results[name] = {
            "total_predictions": scores["total"],
            "top1": scores[1],
            "top3": scores[3],
            "top5": scores[5],
            "perplexity": perplexity,
        }

    return results


def display_model_comparison(results):
    print("=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)
    print(
        f"{'Model':<28} {'Top-1':>10} {'Top-3':>10} "
        f"{'Top-5':>10} {'Perplexity':>14}"
    )
    print("-" * 78)

    for name, values in results.items():
        print(
            f"{name:<28} "
            f"{values['top1']:>9.2f}% "
            f"{values['top3']:>9.2f}% "
            f"{values['top5']:>9.2f}% "
            f"{values['perplexity']:>14.4f}"
        )

    print("=" * 78)
    print("Note: models are evaluated on the same test corpus.")
    print("Perplexity uses Laplace-smoothed probabilities.")
    print()


def evaluate_autocomplete(test_sentences, bigram_counts, context_counts,
                          trigram_counts, trigram_context_counts,
                          vocabulary):
    """Backward-compatible evaluation for the existing main.py."""
    # Import here to avoid a circular import when main.py imports evaluation.py.
    import re

    def tokenize(text):
        text = text.lower()
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    events = _test_events(test_sentences, tokenize)
    vocab_size = len(vocabulary)
    predictions = []
    targets = []
    log_probs = []

    for tokens, i in events:
        target = tokens[i]
        if i >= 2:
            preds = _predict_trigram_for_eval(
                tokens[i - 2], tokens[i - 1], trigram_counts,
                trigram_context_counts, vocabulary
            )
        else:
            preds = []

        if not preds:
            preds = _predict_bigram_for_eval(
                tokens[i - 1], bigram_counts, context_counts, vocabulary
            )
            probability = _laplace_bigram_probability(
                tokens[i - 1], target, bigram_counts,
                context_counts, vocab_size
            )
        else:
            probability = _laplace_trigram_probability(
                tokens[i - 2], tokens[i - 1], target,
                trigram_counts, trigram_context_counts, vocab_size
            )

        predictions.append(preds[:5])
        targets.append(target)
        log_probs.append(math.log(probability))

    scores = _accuracy_for_predictions(predictions, targets)
    perplexity = math.exp(-sum(log_probs) / len(log_probs)) if log_probs else float("inf")

    return {
        "total_predictions": scores["total"],
        "top1": scores[1],
        "top3": scores[3],
        "top5": scores[5],
        "perplexity": perplexity,
    }


def display_evaluation(results):
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Total test predictions: {results['total_predictions']}")
    print(f"Top-1 Accuracy : {results['top1']:.2f}%")
    print(f"Top-3 Accuracy : {results['top3']:.2f}%")
    print(f"Top-5 Accuracy : {results['top5']:.2f}%")
    print(f"Perplexity     : {results['perplexity']:.4f}")
    print()
