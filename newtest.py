import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer



a = np.arange(100)
a = a.reshape(1, 10, -1)[0]

tfidf = TfidfTransformer()
tfidf.fit(a)


b = tfidf.transform(a)
b = b.todense()

print b
