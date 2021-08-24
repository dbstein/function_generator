import numpy as np
import numba
from scipy.linalg import lu_factor, lu_solve
from collections import OrderedDict
from . import new_enough
from .error_models import standard_error_model

# to deal with different numba versions
def jit_it(func, **kwargs):
    kwargs['fastmath'] = True
    if new_enough:
        kwargs['inline'] = 'always'
    return numba.njit(func, **kwargs)

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

def _bisect_search(x, ordered_array):
    n1 = 0
    n2 = ordered_array.size
    while n2 - n1 > 1:
        m = n1 + (n2 - n1) // 2
        if x < ordered_array[m]:
            n2 = m
        else:
            n1 = m
    return n1
bisect_search = jit_it(_bisect_search)

def _bisect_search_lookup(x, ordered_array, bounds_table, idiv):
    table_index = int((x-ordered_array[0]) * idiv)
    n1, n2 = bounds_table[table_index]
    while n2 - n1 > 1:
        m = n1 + (n2 - n1) // 2
        if x < ordered_array[m]:
            n2 = m
        else:
            n1 = m
    return n1
bisect_search_lookup = jit_it(_bisect_search_lookup)

def _numba_chbevl(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x
numba_chbevl = jit_it(_numba_chbevl)

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
numba_chbevl8 = jit_it(_numba_chbevl8)

def _numba_eval(x, lbs, ubs, bounds_table, cs, idiv):
    ind = bisect_search_lookup(x, lbs, bounds_table, idiv)
    a = lbs[ind]
    b = ubs[ind]
    _x = 2*(x-a)/(b-a) - 1.0
    return numba_chbevl(_x, cs[ind])
numba_eval = jit_it(_numba_eval)

def _numba_eval8(x, lbs, ubs, bounds_table, cs, idiv):
    ind = bisect_search_lookup(x, lbs, bounds_table, idiv)
    a = lbs[ind]
    b = ubs[ind]
    _x = 2*(x-a)/(b-a) - 1.0
    return numba_chbevl8(_x, cs[ind])
numba_eval8 = jit_it(_numba_eval8)

class FunctionGenerator(object):
    """
    This class provides a simple way to construct a fast "function evaluator"
    For 1-D functions defined on an interval
    """
    def __init__(self, f, a, b, tol=1e-10, n=8, mw=1e-15, error_model=standard_error_model, mi=100000, verbose=False):
        """
        f:            function to create evaluator for
        a:            lower bound of evaluation interval
        b:            upper bound of evaluation interval
        tol:          accuracy to recreate function to
        n:            degree of chebyshev polynomials to be used
        mw:           minimum width of interval (accuracy no longer guaranteed!)
        error_model: function of chebyshev coefs that gives how much error there is
        mi:          maximum number of intervals
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
        self.mi = mi
        self.error_model = error_model
        self.verbose = verbose
        self.lbs = []
        self.ubs = []
        self.coefs = []
        self.cumulative_integral = []
        _x, _ = get_chebyshev_nodes(-1, 1, self.n)
        self.V = np.polynomial.chebyshev.chebvander(_x, self.n-1)
        self.VLU = lu_factor(self.V)

        self.count = 0

        self._fit(self.a, self.b)
        self.lbs = np.array(self.lbs)
        self.ubs = np.array(self.ubs)
        self.coef_mat = np.row_stack(self.coefs)

        depth = 2**11
        self.bounds_table = np.zeros([depth, 2], dtype=np.int)
        x0, xh = np.linspace(self.a, self.b, depth+1, retstep=True)
        self.bounds_table[0][0] = 0
        ll = len(self.lbs)
        for i in range(1,depth):
            ii = int(bisect_search(x0[i], self.lbs))
            self.bounds_table[i]  [0] = ii
            self.bounds_table[i-1][1] = ii+1
        self.bounds_table[-1][1] = int(bisect_search(x0[i], self.lbs)) + 1
        self.div = xh
        # need to add this so that we don't throw a segfault at the very right endpoint!
        self.bounds_table = np.row_stack([ self.bounds_table, (ll-1, ll) ])

        # calculate cumulative integral for each sub-interval
        int_vec = np.empty(shape=self.n)
        upper_correction = np.empty(shape=self.n)
        upper_correction[0], upper_correction[1] = 1, 0.5
        int_vec[0] = 2
        int_vec[1] = 0
        for i in range(2, self.n):
            denom = float(i**2 - 1)
            int_vec[i] = -(1+(-1)**i) / denom
            upper_correction[i] = -1 / denom
        self.upper_correction = upper_correction

        self.cumulative_integral = 0.5 * np.cumsum(np.sum((int_vec * self.coef_mat), axis=1) * (self.ubs - self.lbs))

        # package things up for numba
        lbs = self.lbs
        ubs = self.ubs
        bounds_table = self.bounds_table
        coef_mat = self.coef_mat
        idiv = 1.0/self.div

        if self.n == 8:
            def _core(x):
                return numba_eval8(x, lbs, ubs, bounds_table, coef_mat, idiv)
            _core = jit_it(_core)
            self._core = _core
        else:
            def _core(x):
                return numba_eval(x, lbs, ubs, bounds_table, coef_mat, idiv)
            _core = jit_it(_core)
            self._core = _core

        def _core_check(x):
            ok = x >= lbs[0] and x <= ubs[-1]
            return _core(x) if ok else np.nan
        _core_check = jit_it(_core_check)
        self._core_check = _core_check

        def _multi_eval(xs, out):
            for i in numba.prange(xs.size):
                out[i] = _core(xs[i])
        _multi_eval = jit_it(_multi_eval, parallel=True)
        self._multi_eval = _multi_eval

        def _multi_eval_check(xs, out):
            for i in numba.prange(xs.size):
                out[i] = _core_check(xs[i])
        _multi_eval_check = jit_it(_multi_eval_check, parallel=True)
        self._multi_eval_check = _multi_eval_check

    def calc_local_int(self, x, segment):
        Tn = np.empty(shape=self.n)
        int_Tn = np.empty(shape=self.n)
        a = self.lbs[segment]
        b = self.ubs[segment]
        _x = 2*(x-a)/(b-a) - 1.0

        Tn[0], Tn[1] = 1, _x
        int_Tn[0], int_Tn[1] = _x, 0.5 * _x * _x
        for i in range(2, self.n):
            Tn[i] = 2 * _x * Tn[i-1] - Tn[i-2]
            denom = i*i - 1
            numer = (2*_x*_x*(i - 1) - i)*Tn[i-1] - _x*(i - 1)*Tn[i-2]
            int_Tn[i] = numer / denom
        return np.dot(self.coef_mat[segment, :], int_Tn)

    def integrate(self, a, b):
        seg_a = bisect_search_lookup(a, self.lbs, self.bounds_table, 1.0 / self.div)
        seg_b = bisect_search_lookup(b, self.lbs, self.bounds_table, 1.0 / self.div)
        int_a = self.calc_local_int(a, seg_a)
        int_b = self.calc_local_int(b, seg_b)
        lower_eval = 0.5 * (np.dot(self.upper_correction, self.coef_mat[seg_a, :]) - int_a) * (self.ubs[seg_a] - self.lbs[seg_a])
        upper_eval = 0.5 * (np.dot(self.upper_correction, self.coef_mat[seg_b, :]) - int_b) * (self.ubs[seg_b] - self.lbs[seg_b])
        return self.cumulative_integral[seg_b] - self.cumulative_integral[seg_a] - upper_eval + lower_eval

    def __call__(self, x, check_bounds=True, out=None):
        """
        Evaluate function at input x
        """
        if isinstance(x, np.ndarray):
            if out is None: out = np.empty(x.shape, dtype=self.dtype)
            if check_bounds:
                self._multi_eval_check(x.ravel(), out.ravel())
            else:
                self._multi_eval(x.ravel(), out.ravel())
            return out
        else:
            if check_bounds:
                return self._core_check(x)
            else:
                return self._core(x)
    def _fit(self, a, b):
        self.count += 1
        if self.count > self.mi:
            raise Exception("Maximum number of intervals exceeded, try another function, increase 'mi', or increase 'n'.")
        m = (a+b)/2.0
        if self.verbose:
            print('[', a, ',', b, ']')
        _, x = get_chebyshev_nodes(a, b, self.n)
        f = self.f(x)
        coefs = lu_solve(self.VLU, f)
        f = np.concatenate([(self.f(a), self.f(b)), f])
        tail_energy = self.error_model(coefs, f)
        if tail_energy < self.tol or b-a < self.mw:
            self.lbs.append(a)
            self.ubs.append(b)
            self.coefs.append(coefs)
        else:
            self._fit(a, m)
            self._fit(m, b)

    def get_base_function(self, check=True):
        return self._core_check if check else self._core
