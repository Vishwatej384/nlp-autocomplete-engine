text = "I love machine learning"

tokens = text.lower().split()

bigrams = []

for i in range(len(tokens) - 1):
    bigram = (tokens[i], tokens[i + 1])
    bigrams.append(bigram)

print(bigrams)