import numpy as np
import numba
from scipy.linalg import lu_factor, lu_solve
from collections import OrderedDict

################################################################################
# basic utilities

def affine_transformation(xin, min_in, max_in, min_out, max_out):
    ran_in = max_in - min_in
    ran_out = max_out - min_out
    rat = ran_out/ran_in
    return (xin - min_in)*rat + min_out

def get_chebyshev_nodes(lb, ub, order):
    xc = np.cos( (2*np.arange(1, order+1)-1)/(2*order)*np.pi )
    x = affine_transformation(xc[::-1], -1, 1, lb, ub)
    return xc[::-1], x

################################################################################
# numba functions

@numba.njit(fastmath=True)
def bisect_search(x, ordered_array):
    n1 = 0
    n2 = ordered_array.size
    while n2 - n1 > 1:
        m = n1 + (n2 - n1) // 2
        if x < ordered_array[m]:
            n2 = m
        else:
            n1 = m
    return n1

@numba.njit(fastmath=True)
def bisect_search_lookup(x, ordered_array, bounds_table, table_len):
    table_index = int(x / table_len * bounds_table.shape[0])
    n1, n2 = bounds_table[table_index]
    while n2 - n1 > 1:
        m = n1 + (n2 - n1) // 2
        if x < ordered_array[m]:
            n2 = m
        else:
            n1 = m
    return n1

@numba.njit(fastmath=True)
def _numba_chbevl(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x

@numba.njit(parallel=True, fastmath=True)
def _numba_multieval_check(xs, lbs, ubs, bounds_table, cs, out):
    n = xs.size
    for i in numba.prange(n):
        out[i] = _numba_eval_check(xs[i], lbs, ubs, bounds_table, cs)

@numba.njit(parallel=True, fastmath=True)
def _numba_multieval(xs, lbs, ubs, bounds_table, cs, out):
    n = xs.size
    for i in numba.prange(n):
        x = xs[i]
        ind = bisect_search_lookup(x, lbs, bounds_table, ubs[-1] - lbs[0])
        a = lbs[ind]
        b = ubs[ind]
        _x = 2*(x-a)/(b-a) - 1.0
        c = cs[ind]
        out[i] = _numba_chbevl(_x, c)

@numba.njit(fastmath=True)
def _numba_eval_check(x, lbs, ubs, bounds_table, cs):
    if x >= lbs[0] and x <= ubs[-1]:
        return _numba_eval(x, lbs, ubs, bounds_table, cs)
    else:
        return np.nan

@numba.njit(fastmath=True)
def _numba_eval(x, lbs, ubs, bounds_table, cs):
    ind = bisect_search_lookup(x, lbs, bounds_table, ubs[-1] - lbs[0])
    a = lbs[ind]
    b = ubs[ind]
    _x = 2*(x-a)/(b-a) - 1.0
    return _numba_chbevl(_x, cs[ind])

def standard_error_model(coefs):
    return np.abs(coefs[-2:]).max()/max(1, np.abs(coefs[0]))

def relative_error_model(coefs):
    return np.abs(coefs[-2:]).max()/np.abs(coefs[0])

class FunctionGenerator(object):
    """
    This class provides a simple way to construct a fast "function evaluator"
    For 1-D functions defined on an interval
    """
    def __init__(self, f, a, b, tol=1e-10, n=12, mw=1e-15, error_model=standard_error_model, verbose=False):
        """
        f:            function to create evaluator for
        a:            lower bound of evaluation interval
        b:            upper bound of evaluation interval
        tol:          accuracy to recreate function to
        n:            degree of chebyshev polynomials to be used
        mw:           minimum width of interval (accuracy no longer guaranteed!)
        error_model: function of chebyshev coefs that gives how much error there is
        verbose: generate verbose output
        """
        self.f = f
        r = b-a
        a1 = a + r/3
        a2 = a + 2*r/3
        self.dtype = self.f(np.array([a1, a2])).dtype
        self.a = float(a)
        self.b = float(b)
        self.tol = tol
        self.n = n
        self.mw = mw
        self.error_model = error_model
        self.verbose = verbose
        self.lbs = []
        self.ubs = []
        self.coefs = []
        _x, _ = get_chebyshev_nodes(-1, 1, self.n)
        self.V = np.polynomial.chebyshev.chebvander(_x, self.n-1)
        self.VLU = lu_factor(self.V)
        self._fit(self.a, self.b)
        self.lbs = np.array(self.lbs)
        self.ubs = np.array(self.ubs)
        self.coef_mat = np.row_stack(self.coefs)

        self.bounds_table = np.zeros([2048, 2], dtype=np.int)
        for i in range(0, self.bounds_table.shape[0]):
            x0 = self.a + i * (self.b - self.a) / self.bounds_table.shape[0]
            x1 = self.a + (i+1) * (self.b - self.a) / self.bounds_table.shape[0]
            self.bounds_table[i][0] = int(bisect_search(x0, self.lbs))
            self.bounds_table[i][1] = int(bisect_search(x1, self.lbs) + 1)

    def __call__(self, x, check_bounds=True, out=None):
        """
        Evaluate function at input x
        """
        if isinstance(x, np.ndarray):
            if out is None: out = np.empty(x.shape, dtype=self.dtype)
            if check_bounds:
                _numba_multieval_check(x.ravel(), self.lbs, self.ubs, self.bounds_table, self.coef_mat, out.ravel())
            else:
                _numba_multieval(x.ravel(), self.lbs, self.ubs, self.bounds_table, self.coef_mat, out.ravel())
            return out
        else:
            if check_bounds:
                return _numba_eval_check(x, self.lbs, self.ubs, self.bounds_table, self.coef_mat)
            else:
                return _numba_eval(x, self.lbs, self.ubs, self.bounds_table, self.coef_mat)
    def _fit(self, a, b):
        m = (a+b)/2.0
        if self.verbose:
            print('[', a, ',', b, ']')
        _, x = get_chebyshev_nodes(a, b, self.n)
        coefs = lu_solve(self.VLU, self.f(x))
        tail_energy = self.error_model(coefs)
        if tail_energy < self.tol or b-a < self.mw:
            self.lbs.append(a)
            self.ubs.append(b)
            self.coefs.append(coefs)
        else:
            self._fit(a, m)
            self._fit(m, b)

    def get_base_function(self, check=True):
        lbs = self.lbs
        ubs = self.ubs
        bounds_table = self.bounds_table
        cs = self.coef_mat
        if check:
            @numba.njit(fastmath=True)
            def func(x):
                return _numba_eval_check(x, lbs, ubs, bounds_table, cs)
        else:
            @numba.njit(fastmath=True)
            def func(x):
                return _numba_eval(x, lbs, ubs, bounds_table, cs)
        return func

    def get_base_function2(self, check=True):
        lbs = self.lbs
        ubs = self.ubs
        cs = self.coef_mat
        if check:
            return _numba_eval_check
        else:
            return _numba_eval
        return func

