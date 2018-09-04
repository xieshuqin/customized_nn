import collections
import re
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import Function
from ._functions import Scatter, Gather
from torch._six import string_classes, int_classes
from torch.utils.data.dataloader import numpy_type_map

def argsort(seq):
    return tuple(sorted(range(len(seq)), key=seq.__getitem__))

class SortByDevice(Function):

    @staticmethod
    def forward(ctx, *inputs):
        assert all(map(lambda i: i.is_cuda, inputs))
        ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        ctx.sorted_index = argsort(ctx.input_gpus)
        return tuple(map(lambda i: inputs[i], ctx.sorted_index))

    @staticmethod
    def backward(ctx, *grad_outputs):
        assert all(map(lambda i: i.is_cuda, grad_outputs))
        return tuple(map(lambda i: grad_outputs[i], ctx.input_gpus))

def sort_by_device(outputs):
    r"""
    Sort variables from different GPUS into an ordered tuple sorted by the
    GPU numbers.
    """
    error_msg = "outputs must contain tensors, dicts or lists; found {}"

    def sort_map(outputs):
        out = outputs[0]
        elem_type = type(out)
        if isinstance(out, Variable):
            return SortByDevice.apply(*outputs)
        if out is None:
            return None
        if isinstance(out, collections.Sequence):
            return type(out)(map(sort_map, zip(*outputs)))
        elif isinstance(out, collections.Mapping):
            return {key: sort_map([d[key] for d in outputs]) for key in out}
        # The sort map only applies to GPU devices, so if the outputs are not
        # iterable or Tensor, they must be cpu memory. In these case, we can't
        # sort them because we don't have devices_id in cpu. So raise error 
        raise TypeError((error_msg.format(elem_type)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return sort_map(outputs)
    finally:
        sort_map = None
