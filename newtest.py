def mydist(x, y):
    np.sum((x-y)**2)

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree',
            metric='pyfunc', func=mydist)
nbrs.fit(X)
NearestNeighbors(algorithm='ball_tree', leaf_size=30, metric='pyfunc',
         n_neighbors=4, radius=1.0)
>>> nbrs.kneighbors(X)
(array([[  0.,   1.,   5.,   8.],
       [  0.,   1.,   2.,  13.],
       [  0.,   2.,   5.,  25.],
       [  0.,   1.,   5.,   8.],
       [  0.,   1.,   2.,  13.],
       [  0.,   2.,   5.,  25.]]), array([[0, 1, 2, 3],
       [1, 0, 2, 3],
       [2, 1, 0, 3],
       [3, 4, 5, 0],
       [4, 3, 5, 0],
       [5, 4, 3, 0]]))
