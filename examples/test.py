import numpy as np
from function_generator import FunctionGenerator as FG
import time
from scipy.special import struve, y0, hankel1
import numba
import os

cpu_count = int(os.cpu_count()/2)

n = 1000000
approx_range = [1e-10, 1000]
test_range = [1e-10, 999]
tol = 1e-14
order = 16

# functions to test evaluation of
true_funcs = [
    lambda x: y0(x),
    lambda x: hankel1(0, x),
    lambda x: np.log(x),
    lambda x: 1/x**8,
]
true_func_disp = [
    'y0(x)',
    'hankel1(0, x)',
    'np.log(x)',
    '1/x**8',
]

def random_in(n, a, b):
    x = np.random.rand(n)
    x *= (b-a)
    x += a
    return x

xtest = random_in(n-10000, test_range[0], test_range[1])
xtest = np.concatenate([np.linspace(test_range[0], test_range[1], 10000), xtest])

print('\nTesting function generator')
print('    minimum test value is: {:0.2e}'.format(xtest.min()))
print('    maximum test value is: {:0.2e}'.format(xtest.max()))

for func, disp in zip(true_funcs, true_func_disp):
    print('\n    Function is: ', disp)

    # test scipy function
    st = time.time()
    ft = func(xtest)
    true_func_time = time.time() - st

    # test approximation function without checks
    st = time.time()
    approx_func = FG(func, approx_range[0], approx_range[1], tol, order, verbose=False)
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

    print('        Error (absolute):                   {:0.1e}'.format(aerr.max()))
    print('        Error (relative):                   {:0.1e}'.format(rerr1.max()))
    print('        Error (large relative):             {:0.1e}'.format(rerr2.max()))
    print('        Error (large relative, checks):     {:0.1e}'.format(rerr2_checks.max()))
    print('        Error (large relative, recompiled): {:0.1e}'.format(rerr2_recompile.max()))
    print('        Scipy time (ms):                    {:0.1f}'.format(true_func_time*1000))
    print('        Build time (ms):                    {:0.1f}'.format(build_time*1000))
    print('        Approx time (ms):                   {:0.1f}'.format(approx_func_time1*1000))
    print('        Approx time, with checks (ms):      {:0.1f}'.format(approx_func_time2*1000))
    print('        Approx time, recompiled  (ms):      {:0.1f}'.format(approx_func_time3*1000))
    print('        Points/Sec/Core, Millions:          {:0.1f}'.format(n/approx_func_time1/1000000/cpu_count))
