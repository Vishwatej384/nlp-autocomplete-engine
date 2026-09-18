# ============================================================
# NLP AUTOCOMPLETE ENGINE - EVALUATION
# ============================================================

import math


def evaluate_autocomplete(
    test_sentences,
    bigram_counts,
    context_counts,
    trigram_counts,
    trigram_context_counts,
    vocabulary,
    autocomplete_function
):
    """
    Evaluate the complete autocomplete system.

    Metrics:
        Top-1 Accuracy
        Top-3 Accuracy
        Top-5 Accuracy
        Perplexity

    The evaluator uses the same:
        Trigram -> Bigram fallback
    strategy as the actual autocomplete system.
    """

    total_predictions = 0

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    log_probability_sum = 0.0
    perplexity_words = 0

    skipped_oov = 0

    # --------------------------------------------------------
    # TEST EVERY SENTENCE
    # --------------------------------------------------------

    for sentence in test_sentences:

        tokens = autocomplete_function.__globals__["tokenize"](
            sentence
        )

        # Need at least 2 words
        if len(tokens) < 2:
            continue

        # ----------------------------------------------------
        # CREATE PREDICTION TASKS
        # ----------------------------------------------------

        for i in range(1, len(tokens)):

            actual_word = tokens[i]

            context_tokens = tokens[:i]

            # Actual word must be known to the model
            if actual_word not in vocabulary:
                skipped_oov += 1
                continue

            context = " ".join(context_tokens)

            # ------------------------------------------------
            # GET TOP-5 PREDICTIONS
            # ------------------------------------------------

            predictions = autocomplete_function(
                context,
                bigram_counts,
                context_counts,
                trigram_counts,
                trigram_context_counts,
                vocabulary,
                top_n=5,
                use_smoothing=False
            )

            if not predictions:
                continue

            predicted_words = [
                word
                for word, probability in predictions
            ]

            total_predictions += 1

            # ------------------------------------------------
            # TOP-1
            # ------------------------------------------------

            if actual_word in predicted_words[:1]:
                top1_correct += 1

            # ------------------------------------------------
            # TOP-3
            # ------------------------------------------------

            if actual_word in predicted_words[:3]:
                top3_correct += 1

            # ------------------------------------------------
            # TOP-5
            # ------------------------------------------------

            if actual_word in predicted_words[:5]:
                top5_correct += 1

            # ------------------------------------------------
            # PERPLEXITY
            # ------------------------------------------------

            # Use the same context strategy as autocomplete:
            #
            # If a Trigram context exists, use Trigram.
            # Otherwise use Bigram.

            probability = None

            if len(context_tokens) >= 2:

                first_word = context_tokens[-2]
                second_word = context_tokens[-1]

                trigram_context = (
                    first_word,
                    second_word
                )

                if trigram_context in trigram_context_counts:

                    probability = (
                        trigram_counts.get(
                            (
                                first_word,
                                second_word,
                                actual_word
                            ),
                            0
                        ) + 1
                    ) / (
                        trigram_context_counts[
                            trigram_context
                        ] + len(vocabulary)
                    )

            # ------------------------------------------------
            # BIGRAM FALLBACK
            # ------------------------------------------------

            if probability is None:

                last_word = context_tokens[-1]

                probability = (
                    bigram_counts.get(
                        (
                            last_word,
                            actual_word
                        ),
                        0
                    ) + 1
                ) / (
                    context_counts.get(
                        last_word,
                        0
                    ) + len(vocabulary)
                )

            # Safety check
            if probability > 0:

                log_probability_sum += math.log(
                    probability
                )

                perplexity_words += 1

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if total_predictions > 0:

        top1_accuracy = (
            top1_correct / total_predictions
        )

        top3_accuracy = (
            top3_correct / total_predictions
        )

        top5_accuracy = (
            top5_correct / total_predictions
        )

    else:

        top1_accuracy = 0.0
        top3_accuracy = 0.0
        top5_accuracy = 0.0

    # --------------------------------------------------------
    # PERPLEXITY
    # --------------------------------------------------------

    if perplexity_words > 0:

        perplexity = math.exp(
            -log_probability_sum
            / perplexity_words
        )

    else:

        perplexity = float("inf")

    return {
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy,
        "perplexity": perplexity,
        "total_predictions": total_predictions,
        "skipped_oov": skipped_oov
    }


def display_evaluation(results):

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        "Total test predictions:",
        results["total_predictions"]
    )

    print(
        "Skipped unknown test words:",
        results["skipped_oov"]
    )

    print()

    print(
        f"Top-1 Accuracy : "
        f"{results['top1_accuracy']:.2%}"
    )

    print(
        f"Top-3 Accuracy : "
        f"{results['top3_accuracy']:.2%}"
    )

    print(
        f"Top-5 Accuracy : "
        f"{results['top5_accuracy']:.2%}"
    )

    if results["perplexity"] != float("inf"):

        print(
            f"Perplexity     : "
            f"{results['perplexity']:.4f}"
        )

    else:

        print(
            "Perplexity     : Not available"
        )

    print("=" * 60)