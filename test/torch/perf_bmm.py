import time
import sys

import torch
from pykeops.torch import Genred

torch.set_num_threads(1)

device = sys.argv[1]

keops_backend = 'CPU'
if device == 'cuda':
    keops_backend = 'GPU'

def timeit(func, it):
    times = []
    for i in range(it):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return sum(times)/it

formula = "TensorDot(a, b, Ind(2,2), Ind(2,2), Ind(1), Ind(0))"
alias = ["a=Vi(4)", "b=Vi(4)"]
keops_bmm = Genred(formula, alias, reduction_op='Sum', axis=1)

N = 1000000
A = torch.rand(N, 2, 2, device=device)
B = torch.rand(N, 2, 2, device=device)
it = 1000

print("torch.bmm() = torch.einsum() :", torch.allclose(torch.bmm(A, B), torch.einsum('nik, nkj->nij', A, B)))
print("torch.einsum() = keops_bmm() :", torch.allclose(torch.einsum('nik, nkj->nij', A, B), keops_bmm(A.view(-1, 4), B.view(-1, 4), backend=keops_backend).view(-1, 2, 2)))


print("torch.bmm(): ", timeit(lambda: torch.bmm(A, B), it))
print("torch.einsum(): ", timeit(lambda: torch.einsum('nik, nkj->nij', A, B), it))
print("keops_bmm(): ", timeit(lambda: keops_bmm(A.view(-1, 4), B.view(-1, 4), backend=keops_backend).view(-1, 2, 2), it))

