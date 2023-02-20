# hook的作用是通过系统来维护一个链表，使得用户拦截（获取）通信消息，用于处理事件
# pytorch中包含forward和backward两个钩子注册函数

"""
源码如下：
import torch


a = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
a.register_forward_hook()

def register_forward_hook(self, hook: Callable[..., None]) -> RemovableHandle:
    r'''Registers a forward hook on the module.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::
    作用：获取forward过程中每层的输入和输出，用于对比hook是不是正确记录

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    '''
    handle = hooks.RemovableHandle(self._forward_hooks)
    self._forward_hooks[handle.id] = hook
    return handle
"""

import torch
from torchvision.models import resnet34

model = resnet34(pretrained=True)
model.eval()
model = model.cuda()

