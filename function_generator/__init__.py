import numba
from packaging import version
new_enough = version.parse(numba.__version__) >= version.parse('0.46.0')
if new_enough:
    from .function_generator_inline import FunctionGenerator
else:
    from .function_generator import FunctionGenerator
from .error_models import standard_error_model, relative_error_model
