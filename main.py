from __future__ import annotations

from sklearn.pipeline import make_pipeline

from ml_preprocess_tools import nlp


if __name__ == "__main__":
    preprocessor = make_pipeline(
        nlp.tokenizer.SpacyTokenizer("en"),
        nlp.normalizer.BaseFormNormalizer(),
        nlp.normalizer.LowerNormalizer(),
        nlp.normalizer.NumberNormalizer())

    vocab = nlp.vocabulary.Vocabulary()


    texts = [
        "This is a simple example of this library.",
        "All the processors are supported scikit-learn API.",
        "Normalization based on POS is available in 8 languages."]

    processed = preprocessor.fit_transform(texts)
    for s in processed:
        print(" ".join(t.surface for t in s))

    vocab = vocab.fit(processed)
    X = vocab.transform(processed)
    print(X)
