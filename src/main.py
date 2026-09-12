sentences = [
    "I like cats",
    "I like dogs",
    "I like pizza",
    "I like cats"
]

bigram_counts = {}
context_counts = {}

for sentence in sentences:
    tokens = sentence.lower().split()

    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i + 1]

        bigram = (current_word, next_word)

        # Count the bigram
        if bigram not in bigram_counts:
            bigram_counts[bigram] = 0

        bigram_counts[bigram] += 1

        # Count the context word
        if current_word not in context_counts:
            context_counts[current_word] = 0

        context_counts[current_word] += 1

print("Bigram counts:")
print(bigram_counts)

print()

print("Context counts:")
print(context_counts)