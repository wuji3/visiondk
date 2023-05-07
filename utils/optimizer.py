from torch.optim import SGD, Adam
from typing import Callable
from functools import wraps

OPTIMIZER = {}

def register_method(fn: Callable):
    key = fn.__name__
    if key in OPTIMIZER:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    OPTIMIZER[key] = fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

@register_method
def sgd(*args, **kwargs):
    return SGD(*args, **kwargs)

@register_method
def adam(*args, **kwargs):
    return Adam(*args, **kwargs)

def create_Optimizer(optimizer: str, lr: float, weight_decay, momentum, params):
    # return partial(OPTIMIZER[optimizer], lr = lr, weight_decay = weight_decay, momentum = momentum)
    return OPTIMIZER[optimizer](params = params, lr = lr, weight_decay = weight_decay, momentum = momentum)

def list_optimizers():
    optimizers = [k for k, v in OPTIMIZER.items()]
    return sorted(optimizers)