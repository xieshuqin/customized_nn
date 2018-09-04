import torch
from torch.nn import Module
from torch.autograd import Variable
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from .sort_by_device import sort_by_device
from .sort_and_gather_except import sort_and_gather_except


class DataParallel(Module):
    r"""Implements data parallelism at the module level.

    IMPORTANT: This is a customized data parallelism that allows not gathering
    all outputs into one device and the input to be directly applied to
    multiple devices.
    This is inspired by the need of intermidiate feature, where a base network
    extracts useful feature that will be used by some other networks multiple
    times. And we have some complex training(e.g. iterative training between
    several networks). Under this setting, we face a lot of redunctant
    scattering and gathering, taking up huge amount of memory on the output
    device. Therefore, we hope there is a machinism that allows us to avoid
    the reduncdant calculation. We can directly return the outputs without
    gathering and directly apply the inputs without scattering.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    .. warning::
        Forward and backwrad hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
        cpu_keywords: list of argument keywords that could be used in `forward` to
            indicating not moving the argument to gpu. Currently, only support
            argument of type: Variable

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0,
                 cpu_keywords=[], minibatch=False, batch_outputs=True,
                 gather_to_one_device=True, sort_and_gather=False,
                 inputs_are_scattered=False, except_keywords=[]):
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])
        self.cpu_keywords = cpu_keywords
        self.minibatch = minibatch
        self.batch_outputs = batch_outputs

        self.gather_to_one_device = gather_to_one_device # gather output to one device
        self.inputs_are_scattered = inputs_are_scattered  
        self.except_keywords = except_keywords  # keywords to not gather output
        self.sort_and_gather = sort_and_gather

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        if self.inputs_are_scattered:
            # input is scattered, then check it is sorted by the GPU ids
            inputs_list, kwargs_list = [], []
            for i, device_id in enumerate(self.device_ids):
                mini_inputs = tuple(x[i] for x in inputs)
                mini_kwargs = dict([(k, v[i]) for k, v in kwargs.items()])
                inputs_list.append(mini_inputs)
                kwargs_list.append(mini_kwargs)
            inputs, kwargs = tuple(inputs_list), tuple(kwargs_list)
        else:
            # input is not scatter, so scatter it
            if self.minibatch:
                inputs_list, kwargs_list = [], []
                for i, device_id in enumerate(self.device_ids):
                    mini_inputs = [x[i] for x in inputs]
                    mini_kwargs = dict([(k, v[i]) for k, v in kwargs.items()])
                    a, b = self._minibatch_scatter(device_id, *mini_inputs, **mini_kwargs)
                    inputs_list.append(a)
                    kwargs_list.append(b)
                inputs = inputs_list
                kwargs = kwargs_list
            else:
                kwargs_cpu = {}
                for k in kwargs:
                    if k in self.cpu_keywords:
                        v = kwargs[k]
                        kwargs_cpu[k] = v
                for k in self.cpu_keywords:
                    kwargs.pop(k, None)
                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                # Split cpu Variables
                for k, v in kwargs_cpu.items():
                    split_size = v.size(self.dim) / len(self.device_ids)
                    assert split_size.is_integer()
                    kwargs_cpu[k] = list(map(Variable, torch.split(v.data, int(split_size), self.dim)))
                kwargs_cpu = list(map(dict, zip(*[[(k, v) for v in vs] for k, vs in kwargs_cpu.items()]))) # a list of dict
                # Merge cpu kwargs with gpu kwargs
                for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
                    d_gpu.update(d_cpu)

        if len(self.device_ids) == 1:
            outputs = [self.module(*inputs[0], **kwargs[0])]
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)

        if self.gather_to_one_device:
            # gather outputs to one device
            if self.sort_and_gather or len(self.except_keywords) > 0:
                # First sort everything. Then if we have except keywords, gather
                # all but the except_keywords
                return self.sort_and_gather_except(outputs, self.output_device, self.except_keywords)
            elif self.batch_outputs:
                return self.gather(outputs, self.output_device)
            else:
                return [self.gather([x], self.output_device) for x in outputs]
        else:
            # not gathering output, just sort them and return
            return self.sort_by_device(outputs)

    def _minibatch_scatter(self, device_id, *inputs, **kwargs):
        kwargs_cpu = {}
        for k in kwargs:
            if k in self.cpu_keywords:
                kwargs_cpu[k] = kwargs[k]
        for k in self.cpu_keywords:
            kwargs.pop(k, None)
        inputs, kwargs = self.scatter(inputs, kwargs, [device_id])
        kwargs_cpu = [kwargs_cpu] # a list of dict
        # Merge cpu kwargs with gpu kwargs
        for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
            d_gpu.update(d_cpu)
        return inputs[0], kwargs[0]

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def sort_by_device(self, outputs):
        return sort_by_device(outputs)

    def sort_and_gather_except(self, outputs, output_device, except_keywords):
        return sort_and_gather_except(outputs, output_device, except_keywords=except_keywords, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
