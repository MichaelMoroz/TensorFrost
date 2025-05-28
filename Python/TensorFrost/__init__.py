from .TensorFrost import *

from . import optimizers
from . import regularizers
from . import clipping
from . import random
from . import sort
from .default import *

# def compile(func):
#     def wrapper(*args, **kwargs):
#         print("Before execution")
#         res = func(*args, **kwargs)
#         print("After execution")
#         return res
#     return wrapper