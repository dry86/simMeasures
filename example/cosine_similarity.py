from sklearn.metrics.pairwise import cosine_similarity
X = [[0, 0, 0], [1, 1, 1]]
Y = [[1, 0, 0], [1, 1, 0]]
print(cosine_similarity(X, Y))