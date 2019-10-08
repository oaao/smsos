from operator import itemgetter

from classifiers import CLASSIFIERS, VECTORIZERS
from config import CONFIG
from input import CsvDataInput


class ClassifierScorer:

    def __init__(self, config=CONFIG, c=CLASSIFIERS, v=VECTORIZERS):

        self.data = CsvDataInput(**CONFIG)

        self.train = self.data.train
        self.test  = self.data.test

        self.c = c
        self.v = v

        self.scores = []

    def score_all(self):

        scores = []

        for classifier in self.c:
            for vectorizer in self.v:

                # train
                vectorize_text = vectorizer.fit_transform(self.train.v2)
                classifier.fit(vectorize_text, self.train.v1)

                # score
                result = self._score(classifier, vectorizer, self.test)
                scores.append(result)

        # result is shaped as (classifier, vectorizer, score)
        #     - we sort in descending order by score
        #     - itemgetter outperforms lambdas even for tuple indices
        scores.sort(key=itemgetter(2), reverse=True)

        self.scores = scores


    def _score(self, c, v, test_data):
        vect_text = v.transform(test_data.v2)
        score = c.score(vect_text, test_data.v1)

        result = (
            c.__class__.__name__,
            v.__class__.__name__,
            score
        )

        return result
