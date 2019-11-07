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

@numba.njit(fastmath=True, inline='always')
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

@numba.njit(fastmath=True, inline='always')
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

@numba.njit(fastmath=True, inline='always')
def _numba_chbevl(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x

@numba.njit(fastmath=True, inline='always')
def _numba_chbevl8(x, c):
    x2 = 2*x
    c0 = c[-3] - c[-1]
    c1 = c[-2] + c[-1]*x2
    tmp = c0
    c0 = c[-4] - c1
    c1 = tmp + c1*x2
    tmp = c0
    c0 = c[-5] - c1
    c1 = tmp + c1*x2
    tmp = c0
    c0 = c[-6] - c1
    c1 = tmp + c1*x2
    tmp = c0
    c0 = c[-7] - c1
    c1 = tmp + c1*x2
    tmp = c0
    c0 = c[-8] - c1
    c1 = tmp + c1*x2
    return c0 + c1*x

@numba.njit(fastmath=True, inline='always')
def _numba_eval(x, lbs, ubs, bounds_table, cs):
    ind = bisect_search_lookup(x, lbs, bounds_table, ubs[-1] - lbs[0])
    a = lbs[ind]
    b = ubs[ind]
    _x = 2*(x-a)/(b-a) - 1.0
    return _numba_chbevl(_x, cs[ind])

@numba.njit(fastmath=True, inline='always')
def _numba_eval8(x, lbs, ubs, bounds_table, cs):
    ind = bisect_search_lookup(x, lbs, bounds_table, ubs[-1] - lbs[0])
    a = lbs[ind]
    b = ubs[ind]
    _x = 2*(x-a)/(b-a) - 1.0
    return _numba_chbevl8(_x, cs[ind])

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

        # package things up for numba
        lbs = self.lbs
        ubs = self.ubs
        bounds_table = self.bounds_table
        coef_mat = self.coef_mat

        if self.n == 8:
            @numba.njit(fastmath=True, inline='always')
            def _core(x):
                return _numba_eval8(x, lbs, ubs, bounds_table, coef_mat)
            self._core = _core
        else:
            @numba.njit(fastmath=True, inline='always')
            def _core(x):
                return _numba_eval(x, lbs, ubs, bounds_table, coef_mat)
            self._core = _core

        @numba.njit(fastmath=True, inline='always')
        def _core_check(x):
            ok = x >= lbs[0] and x <= ubs[-1]
            return _core(x) if ok else np.nan
        self._core_check = _core_check

        @numba.njit(parallel=True, fastmath=True)
        def _multi_eval(xs, out):
            for i in numba.prange(xs.size):
                out[i] = _core(xs[i])
        self._multi_eval = _multi_eval

        @numba.njit(parallel=True, fastmath=True)
        def _multi_eval_check(xs, out):
            for i in numba.prange(xs.size):
                out[i] = _core_check(xs[i])
        self._multi_eval_check = _multi_eval_check

    def __call__(self, x, check_bounds=True, out=None):
        """
        Evaluate function at input x
        """
        if isinstance(x, np.ndarray):
            if out is None: out = np.empty(x.shape, dtype=self.dtype)
            if check_bounds:
                self._multi_eval(x.ravel(), out.ravel())
            else:
                self._multi_eval_check(x.ravel(), out.ravel())
            return out
        else:
            if check_bounds:
                return self._core_check(x)
            else:
                return self._core(x)
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
        return self._core_check if check else self._core
