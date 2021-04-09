import os
import sys
import psutil

if len(sys.argv) > 1:
    scores = sys.argv[1]
    os.environ['MKL_NUM_THREADS'] = scores
    os.environ['OMP_NUM_THREADS'] = scores
    os.environ['NUMBA_NUM_THREADS'] = scores
    cpu_count = int(scores)
    thread_count = int(scores)
else:
    cpu_count = psutil.cpu_count(logical=False)
    thread_count = psutil.cpu_count(logical=True)

import numpy as np
from function_generator import FunctionGenerator as FG
from function_generator import standard_error_model, relative_error_model, new_error_model
import time
from scipy.special import k0, struve, y0, hankel1
import numba

n = 1000*1000*10
approx_range = [1e-10, 1000]
test_range = [1e-10, 999]
tol = 1e-14
order = 8

# functions to test evaluation of
true_funcs = [
    lambda x: k0(0.1*x),
    lambda x: k0(0.1*x),
    lambda x: y0(x),
    lambda x: hankel1(0, x),
    lambda x: np.log(x),
    lambda x: 1/x**8,
    lambda x: 1/np.sqrt(x),
    lambda x: x*x,
    lambda x: np.sin(1/x),
]
true_func_disp = [
    'k0(x/10); Using relative error model, relative error should be good',
    'k0(x/10); Using standard error model, relative error not guaranteed',
    'y0(x); Using standard error model, relative error not guaranteed',
    'hankel1(0, x); Using standard error model, relative error not guaranteed',
    'np.log(x); Using standard error model, relative error not guaranteed',
    '1/x**8; Using relative error model, relative error should be good',
    '1/sqrt(x) Using relative error model, relative error should be good',
    'x*x; refinement test does not work; only gives one interval!',
    'sin(1/x); should fail',
]
error_models = [
    relative_error_model,
    standard_error_model,
    standard_error_model,
    standard_error_model,
    standard_error_model,
    relative_error_model,
    relative_error_model,
    new_error_model,
    standard_error_model,
]

def random_in(n, a, b):
    x = np.random.rand(n)
    x *= (b-a)
    x += a
    return x

xtest = random_in(n-10000, test_range[0], test_range[1])
xtest = np.concatenate([np.linspace(test_range[0], test_range[1], 10000), xtest])

print('\nTesting function generator on', n, 'points, using', cpu_count, 'cores,', thread_count, 'threads.')
print('    minimum test value is: {:0.2e}'.format(xtest.min()))
print('    maximum test value is: {:0.2e}'.format(xtest.max()))
print('')
print('    Standard error model means normalization by max(1, value)')
print('    Relative error model means normalization by value')

for func, disp, error_model in zip(true_funcs, true_func_disp, error_models):
    print('\n    Function is: ', disp)

    # test scipy function
    st = time.time()
    ft = func(xtest)
    true_func_time = time.time() - st

    # test approximation function without checks
    st = time.time()
    approx_func = FG(func, approx_range[0], approx_range[1], tol, order, error_model=error_model)
    build_time = time.time() - st
    fa = approx_func(xtest, check_bounds=False)
    out = np.empty(n, dtype=fa.dtype)
    st = time.time()
    fa = approx_func(xtest, check_bounds=False, out=out)
    approx_func_time1 = time.time()-st

    # test approximation function with checks
    fa = approx_func(xtest, check_bounds=True)
    st = time.time()
    fa1 = approx_func(xtest, check_bounds=True)
    approx_func_time2 = time.time()-st

    # extract serial function, and compile it
    base_func = approx_func.get_base_function(check=False)
    @numba.njit(parallel=True, fastmath=True)
    def func_eval(xs, out):
        for i in numba.prange(xs.size):
            out[i] = base_func(xs[i])
    fa2 = np.empty_like(fa)
    func_eval(xtest, fa2)
    st = time.time()
    func_eval(xtest, fa2)
    approx_func_time3 = time.time()-st

    aerr = np.abs(fa-ft)
    rerr1 = np.abs(fa-ft)/np.abs(ft)
    scale = np.abs(ft)
    scale[scale < 1] = 1
    rerr2 = np.abs(fa-ft)/scale
    rerr2_checks = np.abs(fa1-ft)/scale
    rerr2_recompile = np.abs(fa2-ft)/scale

    print('        Error (absolute):                {:0.1e}'.format(aerr.max()))
    print('        Error (relative):                {:0.1e}'.format(rerr1.max()))
    print('        Error (standard):                {:0.1e}'.format(rerr2.max()))
    print('        Error (standard, checks):        {:0.1e}'.format(rerr2_checks.max()))
    print('        Error (standard, recompiled):    {:0.1e}'.format(rerr2_recompile.max()))
    print('        Scipy time (ms):                 {:0.1f}'.format(true_func_time*1000))
    print('        Build time (ms):                 {:0.1f}'.format(build_time*1000))
    print('        Approx time (ms):                {:0.1f}'.format(approx_func_time1*1000))
    print('        Approx time, with checks (ms):   {:0.1f}'.format(approx_func_time2*1000))
    print('        Approx time, recompiled  (ms):   {:0.1f}'.format(approx_func_time3*1000))
    print('        Points/Sec/Core, Millions:       {:0.1f}'.format(n/approx_func_time3/1000000/cpu_count))

