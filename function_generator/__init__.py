import numba
from packaging import version
new_enough = version.parse(numba.__version__) >= version.parse('0.46.0')
from .error_models import standard_error_model, relative_error_model, new_error_model
from .function_generator import FunctionGenerator
