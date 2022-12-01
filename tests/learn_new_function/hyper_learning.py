import types
from argparse import Namespace
from typing import Any, Optional, Union, Sequence, Union

# from pytorch_lightning.utilities.parsing import save_hyperparameters
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin


class ManuallyArgsModel(HyperparametersMixin):
    def __init__(self, arg1, arg2, arg3, arg4, arg5) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        print('forward function')


model = ManuallyArgsModel(1, 'das', 'dasda', 1222, 3333)
print(model.hparams)


class SingleArgModel(HyperparametersMixin):
    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        print('forward function')


model = SingleArgModel(Namespace(p1=1, p2='das', p3='dasdsad'))
print(model.hparams)