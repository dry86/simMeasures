from ranksim import RankSimilarityClassifier
X = [[0, 1], [1, 0]]
y = [0, 1]
clf = RankSimilarityClassifier()
clf.fit(X, y)
pred = clf.predict(X)
print(pred)
