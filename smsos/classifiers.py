# classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# vectorizers
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer
)

CLASSIFIERS = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    BernoulliNB(),
    CalibratedClassifierCV(),
    DecisionTreeClassifier(),
    DummyClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    OneVsRestClassifier(LogisticRegression()),
    OneVsRestClassifier(SVC(kernel='linear')),
    PassiveAggressiveClassifier(),
    RandomForestClassifier(n_estimators=100, n_jobs=-1),
    RidgeClassifier(),
    RidgeClassifierCV(),
    SGDClassifier()
]

VECTORIZERS = [
    CountVectorizer(),
    HashingVectorizer(),
    TfidfVectorizer()
]
