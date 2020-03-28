import torch
from knn_cosine import KnnCosine

x = torch.FloatTensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = torch.FloatTensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
index = KnnCosine().knn(x, y, k=2, loop=False)
print(index)
