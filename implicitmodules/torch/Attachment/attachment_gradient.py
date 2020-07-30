from collections import Iterable

import torch


def compute_attachment_gradient(sources, targets, attachment):
    assert type(sources) == type(targets)
    assert isinstance(sources, Iterable)

    # Terrible way to handle single and list of objects
    islist = True
    if isinstance(sources, torch.Tensor):
        islist = False
        sources = [sources]
        targets = [targets]

    sources = [source.data.requires_grad_() for source in sources]
    distance = sum([attachment(source, target) for source, target in zip(sources, targets)])

    distance.backward()

    # Same here, even worse
    if islist:
        return [source.grad for source in sources]
    else:
        return sources[0].grad.data

