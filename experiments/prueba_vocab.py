with open("corpus/corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Vocabulario
vocab = sorted(set(corpus))
vocab_size = len(vocab)

print(f"Tamaño de corpus: {len(corpus)}")
print(f"Tamaño de vocabulario: {vocab_size}")