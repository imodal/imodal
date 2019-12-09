import torch

def A(self, dim, dtype, device):
    cst = -math.sqrt(2.)/2.
    if dim == 2:
        A = torch.zeros(2, 2, 3, 3, device=device, dtype=dtype)

        # A[0, 0]
        A[0, 0, 0, 0] = 1.
        A[0, 0, 1, 1] = 2.

        # A[1, 1]
        A[1, 1, 1, 1] = 2.
        A[1, 1, 2, 2] = 1.

        # A[0, 1]
        A[0, 1, 0, 1] = cst
        A[0, 1, 1, 0] = cst
        A[0, 1, 1, 2] = cst
        A[0, 1, 2, 1] = cst

        # A[1, 0]
        A[1, 0] = A[0, 1]

        return A
    elif dim == 3:
        A = torch.zeros(3, 3, 6, 6, device=device, dtype=dtype)

        # A[0, 0]
        A[0, 0, 0, 0] = 1.
        A[0, 0, 3, 3] = 2.
        A[0, 0, 4, 4] = 2.

        # A[1, 1]
        A[1, 1, 1, 1] = 1.
        A[1, 1, 3, 3] = 2.
        A[1, 1, 5, 5] = 2.

        # A[2, 2]
        A[2, 2, 2, 2] = 1.
        A[2, 2, 4, 4] = 2.
        A[2, 2, 5, 5] = 2.

        # A[0, 1]
        A[0, 1, 0, 3] = cst
        A[0, 1, 3, 0] = cst
        A[0, 1, 1, 3] = cst
        A[0, 1, 3, 1] = cst
        A[0, 1, 4, 5] = -1.
        A[0, 1, 5, 4] = -1.

        # A[1, 0]
        A[1, 0] = A[0, 1]

        # A[0, 2]
        A[0, 2, 0, 4] = cst
        A[0, 2, 4, 0] = cst
        A[0, 2, 4, 2] = cst
        A[0, 2, 2, 4] = cst
        A[0, 2, 4, 5] = -1.
        A[0, 2, 5, 4] = -1.

        # A[1, 2]
        A[1, 2, 1, 5] = cst
        A[1, 2, 5, 1] = cst
        A[1, 2, 2, 5] = cst
        A[1, 2, 5, 2] = cst
        A[1, 2, 3, 4] = -1.
        A[1, 2, 4, 3] = -1.

        # A[2, 1]
        A[2, 1] = A[1, 2]

        return A
    else:
        raise NotImplementedError

