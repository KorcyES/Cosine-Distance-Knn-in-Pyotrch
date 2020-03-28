import torch
import torch.nn.functional as F


class KnnCosine(object):
    def __init__(self):
        pass

    @staticmethod
    def knn(x, query, k=2, loop=False):
        r"""Finds for each element in :obj:`query` the :obj:`k` nearest points in
            :obj:`x`.
            .. test setup:
            import torch
            import KnnCosine

            .. test code:
            x = torch.FloatTensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            y = torch.FloatTensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            index = KnnCosine().knn(x, y, k=2, loop=False)
            print(index)

            .. result:
            tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                    [2, 1, 3, 0, 3, 0, 2, 1]])
        """
        if torch.cuda.is_available():
            x = x.cuda()
            query = query.cuda()
        if not loop:
        	k += 1
        x = F.normalize(x)
        query = F.normalize(query)
        query_result = torch.matmul(x, query.t())
        dist, col = torch.topk(query_result, k=k, dim=0)
        col = col.t()
        if torch.cuda.is_available():
            col = col.cpu()
        row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
        if not loop:
            mask = row != col
            row, col = row[mask], col[mask]
        return torch.stack([row, col], dim=0)
