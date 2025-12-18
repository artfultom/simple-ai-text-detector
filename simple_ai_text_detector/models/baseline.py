from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    def __init__(self, language="english", remove_stopwords=True, lemmatize=True):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(language))

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)

    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]


class TfidfLogisticClassifier:
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42,
        use_preprocessing: bool = True,
        language: str = "english",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ):
        self.use_preprocessing = use_preprocessing

        if self.use_preprocessing:
            self.preprocessor = TextPreprocessor(
                language=language,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
            )

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.95
        )
        self.classifier = LogisticRegression(
            max_iter=max_iter, C=C, random_state=random_state, class_weight="balanced"
        )

    def fit(self, X_train, y_train):
        if self.use_preprocessing:
            X_train = self.preprocessor.preprocess_batch(X_train)

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)

    def predict(self, X):
        if self.use_preprocessing:
            X = self.preprocessor.preprocess_batch(X)

        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)

    def predict_proba(self, X):
        if self.use_preprocessing:
            X = self.preprocessor.preprocess_batch(X)

        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_tfidf)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
