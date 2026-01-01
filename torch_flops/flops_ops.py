from torch import nn, Tensor, Size
from torch.types import Number
from typing import Optional, Union

__all__ = ['MODULE_FLOPs_MAPPING', 'FUNCTION_FLOPs_MAPPING', 'METHOD_FLOPs_MAPPING']


def flops_zero() -> int:
    return 0


def flops_elemwise(result_shape: Size) -> int:
    return result_shape.numel()


def flops_matmul(tensor1, tensor2) -> int:
    assert tensor1.shape[-1] == tensor2.shape[-2 if len(tensor2.shape) > 1 else 0]
    return 2 * tensor1.numel() * int((tensor2 != 0.).sum().item()) // tensor1.shape[-1]

# For nn.modules.*
def flops_convnd(module: nn.modules.conv._ConvNd, input_shape: Size, result_shape: Size) -> int:
    conv_elements = int((module.weight != 0).sum().item())
    return 2 * conv_elements * (result_shape.numel()//module.out_channels) - int(module.bias is None) * module.groups * result_shape.numel()


def flops_avgpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return kernel_size.numel() * result_shape.numel()


def flops_adaptive_avgpoolnd(module: nn.modules.pooling._AdaptiveAvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return kernel_size.numel() * result_shape.numel()


def flops_maxpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return (kernel_size.numel() - 1) * result_shape.numel()


def flops_adaptive_maxpoolnd(module: nn.modules.pooling._AdaptiveMaxPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return (kernel_size.numel() - 1) * result_shape.numel()


def flops_functional_convnd(bias: Optional[Tensor], groups: int, in_channels: int, out_channels: int, weight: Tensor, result_shape: Size) -> int:
    conv_elements = int((weight != 0).sum().item())
    return 2 * conv_elements * (result_shape.numel()//out_channels) - int(bias is None) * groups * result_shape.numel()


# For ModuleFLOPs
def ModuleFLOPs_zero(module: nn.Linear, result: Tensor, *args, **kwargs) -> int:
    return flops_zero()


def ModuleFLOPs_elemwise(module: nn.Module, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    total_flops = flops_elemwise(result_shape)
    return total_flops


def ModuleFLOPs_LeakyReLU(module: nn.LeakyReLU, result: Tensor, *args, **kwargs) -> int:
    return result.numel() * 4


def ModuleFLOPs_Linear(module: nn.Linear, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    result_shape = result.shape

    total_flops = flops_matmul(args[0], module.weight.T)
    if module.bias is not None:
        total_flops += flops_elemwise(result_shape)

    return total_flops


def ModuleFLOPs_ConvNd(module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_convnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AvgPoolNd(module: Union[nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_avgpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AdaptiveAvgPoolNd(module: Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_adaptive_avgpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_MaxPoolNd(module: Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_maxpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AdaptiveMaxPoolNd(module: Union[nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_adaptive_maxpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_Norm(module: Union[nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)
    assert not module.training, "Only support `eval` mode."

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    # (X-mean)/std
    total_flops = flops_elemwise(input_shape) * 2
    if (hasattr(module, 'affine') and module.affine) or (hasattr(module, 'elementwise_affine'), module.elementwise_affine):
        total_flops += flops_elemwise(input_shape) * 2

    return total_flops


def ModuleFLOPs_GELU(module: nn.GELU, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    total_flops = flops_elemwise(result_shape)
    if module.approximate is None:
        raise NotImplementedError()

    return total_flops


# For FunctionFLOPs
def FunctionFLOPs_zero(result: Tensor, *args, **kwargs) -> int:
    return flops_zero()


def FunctionFLOPs_elemwise(result: Union[Tensor, Number], *args, **kwargs) -> int:
    assert len(args) == 2, len(args)

    total_flops = None
    if isinstance(result, Number):
        total_flops = 1
    elif isinstance(result, Tensor):
        total_flops = flops_elemwise(result.shape)
    else:
        raise TypeError(type(result))

    return total_flops


def FunctionFLOPs_matmul(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 2, len(args)
    tensor_A, tensor_B = args
    assert isinstance(tensor_A, Tensor) and isinstance(tensor_B, Tensor)

    total_flops = flops_matmul(tensor_A, tensor_B)
    return total_flops


def FunctionFLOPs_linear(result: Tensor, *args, **kwargs) -> int:
    if len(args) == 3:
        input, weight, bias = args
    elif len(args) == 2:
        input, weight = args
        bias = kwargs.get('bias')
    else:
        input = args[0]
        weight = kwargs.get('weight')
        bias = kwargs.get('bias')

    assert isinstance(input, Tensor) and isinstance(weight, Tensor)

    total_flops = flops_matmul(input, weight.T)
    if bias is not None:
        total_flops += flops_elemwise(result.shape)
    return total_flops 


def FunctionFLOPs_convnd(result: Tensor, *args, **kwargs) -> int:
    
    input = args[0]
    if len(args) > 1:
        weight = args[1]
    else:
        weight = kwargs.get('weight')

    assert isinstance(input, Tensor)
    assert isinstance(weight, Tensor)

    in_channels = weight.shape[1]
    out_channels = weight.shape[0]
    bias = kwargs.get('bias')
    groups = kwargs.get('groups', None)
    if groups is None:
        groups = 1
    stride = kwargs.get('stride', None)
    if stride is None:
        stride = 1
    padding = kwargs.get('padding', None)
    if padding is None:
        padding = 0
    result_shape = result.shape

    return flops_functional_convnd(bias, groups, in_channels, out_channels, weight, result_shape)

def FunctionFLOPs_leaky_relu(result: Tensor, *args, **kwargs) -> int:
    return result.numel() * 4

def FunctionFLOPs_interpolate(result: Tensor, *args, **kwargs) -> int:
    input = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size = kwargs.get('size', None)

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            prod = 1
            for s in size:
                prod *= s
            return int(prod)
        else:
            return int(size)
    
    if len(args) > 2:
        scale_factor = args[2]
    else:
        scale_factor = kwargs.get('scale_factor', None)

    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        prod = 1
        for s in scale_factor:
            prod *= s
        flops *= int(prod)
    else:
        flops *= scale_factor**len(input)

    return flops


def FunctionFLOPs_scaled_dot_product_attention(result: Tensor, *args, **kwargs) -> int:
    assert len(args) >= 3, len(args)
    q, k, v = args[:3]
    assert isinstance(q, Tensor) and isinstance(k, Tensor) and isinstance(v, Tensor)

    q_shape, k_shape, v_shape = q.shape, k.shape, v.shape
    batch_heads = 1
    for dim in q_shape[:-2]:
        batch_heads *= dim
    Lq, d = q_shape[-2], q_shape[-1]
    Lk = k_shape[-2]
    dv = v_shape[-1]

    flops_qk = (2 * d - 1) * batch_heads * Lq * Lk
    flops_scale = batch_heads * Lq * Lk

    attn_mask = args[3] if len(args) > 3 else kwargs.get('attn_mask')
    is_causal = args[5] if len(args) > 5 else kwargs.get('is_causal', False)
    flops_mask = batch_heads * Lq * Lk if (attn_mask is not None or is_causal) else 0

    flops_softmax = batch_heads * Lq * Lk  # exp
    flops_softmax += batch_heads * Lq * (Lk - 1)  # sum
    flops_softmax += batch_heads * Lq * Lk  # div

    dropout_p = args[4] if len(args) > 4 else kwargs.get('dropout_p', 0.0)
    flops_dropout = batch_heads * Lq * Lk if (dropout_p and dropout_p > 0) else 0

    flops_av = (2 * Lk - 1) * batch_heads * Lq * dv

    total_flops = flops_qk + flops_scale + flops_mask + flops_softmax + flops_dropout + flops_av
    return int(total_flops)


# For MethodFLOPs
def MethodFLOPs_zero(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    return flops_zero()


def MethodFLOPs_elemwise(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    return flops_elemwise(result.shape)


def MethodFLOPs_sum(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    this_shape = self_obj.squeeze().shape
    result_shape = result.squeeze().shape

    total_flops = None
    if len(result_shape) == 0:
        total_flops = self_obj.numel() - 1
    else:
        kept_shape = list(this_shape)
        for s in result_shape:
            kept_shape.remove(s)
        kept_shape = Size(kept_shape)
        total_flops = kept_shape.numel() * (result_shape.numel() - 1)

    return total_flops


def MethodFLOPs_softmax(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    this_shape = self_obj.shape
    result_shape = result.shape
    assert this_shape == result_shape

    exp_flops = flops_elemwise(this_shape)

    dim_reduce: int = args_tail[0] if args_tail else kwargs.get('dim')
    dims_kept = list(this_shape)
    dims_kept.pop(dim_reduce)
    dims_kept = Size(dims_kept)
    sum_flops = (this_shape[dim_reduce] - 1) * dims_kept.numel()

    div_flops = flops_elemwise(this_shape)

    total_flops = exp_flops + sum_flops + div_flops
    return total_flops
    


MODULE_FLOPs_MAPPING = {
    'Linear': ModuleFLOPs_Linear,
    'Identity': ModuleFLOPs_zero,
    'Conv1d': ModuleFLOPs_ConvNd,
    'Conv2d': ModuleFLOPs_ConvNd,
    'Conv3d': ModuleFLOPs_ConvNd,
    'AvgPool1d': ModuleFLOPs_AvgPoolNd,
    'AvgPool2d': ModuleFLOPs_AvgPoolNd,
    'AvgPool3d': ModuleFLOPs_AvgPoolNd,
    'AdaptiveAvgPool1d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'AdaptiveAvgPool2d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'AdaptiveAvgPool3d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'MaxPool1d': ModuleFLOPs_MaxPoolNd,
    'MaxPool2d': ModuleFLOPs_MaxPoolNd,
    'MaxPool3d': ModuleFLOPs_MaxPoolNd,
    'AdaptiveMaxPool1d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'AdaptiveMaxPool2d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'AdaptiveMaxPool3d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'LayerNorm': ModuleFLOPs_Norm,
    'BatchNorm1d': ModuleFLOPs_Norm,
    'BatchNorm2d': ModuleFLOPs_Norm,
    'BatchNorm3d': ModuleFLOPs_Norm,
    'InstanceNorm1d': ModuleFLOPs_Norm,
    'InstanceNorm2d': ModuleFLOPs_Norm,
    'InstanceNorm3d': ModuleFLOPs_Norm,
    'GroupNorm': ModuleFLOPs_Norm,
    'Dropout': ModuleFLOPs_zero,
    'GELU': ModuleFLOPs_GELU,
    'ReLU': ModuleFLOPs_elemwise,
    'Flatten': ModuleFLOPs_zero,
    'LeakyReLU': ModuleFLOPs_LeakyReLU,
    'type_as': ModuleFLOPs_zero
}
FUNCTION_FLOPs_MAPPING = {
    'getattr': FunctionFLOPs_zero,
    'getitem': FunctionFLOPs_zero,
    'mul': FunctionFLOPs_elemwise,
    'truediv': FunctionFLOPs_elemwise,
    'sub': FunctionFLOPs_elemwise,
    'matmul': FunctionFLOPs_matmul,
    'add': FunctionFLOPs_elemwise,
    'concat': FunctionFLOPs_zero,
    '_assert': FunctionFLOPs_zero,
    'eq': FunctionFLOPs_elemwise,
    'cat': FunctionFLOPs_zero,
    'linear': FunctionFLOPs_linear,
    'conv1d': FunctionFLOPs_convnd,
    'conv2d': FunctionFLOPs_convnd,
    'conv3d': FunctionFLOPs_convnd,
    'leaky_relu': FunctionFLOPs_leaky_relu,
    'pad': FunctionFLOPs_zero,
    'floordiv': FunctionFLOPs_zero,
    'flip': FunctionFLOPs_zero,
    'interpolate': FunctionFLOPs_interpolate,
    'scaled_dot_product_attention': FunctionFLOPs_scaled_dot_product_attention,
}
METHOD_FLOPs_MAPPING = {
    'reshape': MethodFLOPs_zero,
    'permute': MethodFLOPs_zero,
    'unbind': MethodFLOPs_zero,
    'transpose': MethodFLOPs_zero,
    'repeat': MethodFLOPs_zero,
    'unsqueeze': MethodFLOPs_zero,
    'exp': MethodFLOPs_elemwise,
    'sum': MethodFLOPs_sum,
    'div': MethodFLOPs_elemwise,
    'softmax': MethodFLOPs_softmax,
    'expand': MethodFLOPs_zero,
    'flatten': MethodFLOPs_zero,
    'view': MethodFLOPs_zero,
    'cuda': MethodFLOPs_zero,
    'flip': MethodFLOPs_zero,
    'type_as': MethodFLOPs_zero,
    'size': MethodFLOPs_zero,
    'clone': MethodFLOPs_zero,
    'new_empty': MethodFLOPs_zero,
    'normal_': MethodFLOPs_zero,
    'pow': MethodFLOPs_zero,
}
