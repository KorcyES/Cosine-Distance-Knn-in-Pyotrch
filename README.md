### Cosine Distance K-Nearest Neighbor In Pytorch

Using cuda to speed up Knn calculations.

Input x: $N_1 \times F$ torch tensor query: $N_2 \times F$ torch tensor, where $N_1$, $N_2$ is the number of vector and F is the dimension of feature.

Output: $2 \times M$ torch tensor, where $M$ is the number of edges[$M = N_2 \times k$].

It takes only 50 seconds to calculate the 25 nearest neighbors of [39930, 2048] on Titan X, but it requires nearly 8G of memory.

