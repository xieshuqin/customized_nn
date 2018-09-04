import collections
import re
import numpy as np
import torch
from torch.autograd import Variable
from ._functions import Gather
from torch._six import string_classes, int_classes
from torch.utils.data.dataloader import numpy_type_map
from .sort_by_device import sort_by_device, SortByDevice


def sort_and_gather_except(outputs, target_device, except_keywords=[], dim=0):
    r"""
    First sort the outputs according to their gpu device id. Then if the output
    is in the except_keywords, just return the sorted tuple. Otherwise, gather
    them on a specified device.
      (-1 means the CPU).
    """
    error_msg = "outputs must contain tensors, numbers, dicts or lists; found {}"

    def sort_and_gather_except_map(outputs):
        out = outputs[0]
        elem_type = type(out)
        if isinstance(out, Variable):
            return Gather.apply(target_device, dim, *(SortByDevice.apply(*outputs)))
        if out is None:
            return None
        if isinstance(out, collections.Sequence):
            return type(out)(map(sort_and_gather_except_map, zip(*outputs)))
        elif isinstance(out, collections.Mapping):
            return {key: sort_by_device([d[key] for d in outputs]) if key in except_keywords \
                else sort_and_gather_except_map([d[key] for d in outputs]) for key in out}
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = out
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return Variable(torch.from_numpy(np.concatenate(outputs, dim)))
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return Variable(numpy_type_map[elem.dtype.name](list(map(py_type, outputs))))
        elif isinstance(out, int_classes):
            return Variable(torch.LongTensor(outputs))
        elif isinstance(out, float):
            return Variable(torch.DoubleTensor(outputs))
        elif isinstance(out, string_classes):
            return outputs

        raise TypeError((error_msg.format(elem_type)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return sort_and_gather_except_map(outputs)
    finally:
        sort_and_gather_except_map = None
