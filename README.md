# function_generator: Fast, compilable recreation for slow functions

Special functions are typically slow and have implementations that are not compatible with numba's jit compiler.  This package constructs *approximations* of functions with tunable accuracy, and provides functions for rapidly evaluating these approximations which are compatible with numba. The approximations are built with Chebyshev series, and the domain is subdivided where needed; providing stable approximations that can be accurate to nearly machine precision. For functions where the number of approximation intervals needed is relatively small, the speed of evaluating the approximation can approach native functions; for example an approximation of np.log valid on the interval [1e-10, 1000] is as fast at evaluation as the Intel SVML implementaiton of log.

Constructing a function to approximate a given function on the interval [a,b] is as easy as:
```python
approx_func = FG(func, a, b)
```
By default, the function uses eighth order Chebyshev expansions and sets the tolerance to 10 digits; these are all configurable. This function may now be called on a scalar or ndarray as:
```python
y = approx_func(x)
```
or, for ndarrays, to save time in allocating space for the result:
```python
approx_func(x, y)
```
Moreover, the core scalar evaluation function may be accessed as:
```python
core_func = approx_func.get_base_function(check)
```
where check is a bool that determines whether the core_func checks whether x is in the approximation interval [a,b].  The checked function returns np.nan outside of the approximation interval; the unchecked function returns garbage. This function is numba compatible, thus one may define functions overtop of it:
```python
@numba.njit(fastmath=True)
def kernel(sx, sy, tx, ty):
    dx = tx - sx
    dy = ty - sy
    d = np.sqrt(dx*dx + dy*dy)
    return core_func(d)
```
For numba versions that support the 'inline' argument to njit (> 0.46.0), the returned core_func has been set to always inline, so that performance is maintained in more complicated functions using core_func.

## Known issues:

1. The given error_models, and how refinement is triggered, is not completely robust, particularly when the interval is very large and the function is smooth and varies dramatically in scale. For example, the function `f(x)=x*x` fails miserably on the approximation range [1e-10, 1000]; as it triggers no refinement and only uses one approximation interval. The function should be made robust to this problem.
2. When the function reaches the maximum number of intervals (user specifiable as the argument `mi`), the error traces back through the whole recursion, making for pretty messy error output. Could this be neatened?

## Results on my mac laptop:

```python
Testing function generator on 10000000 points, using 4 cores, 8 threads.
    minimum test value is: 1.00e-10
    maximum test value is: 9.99e+02

    Standard error model means normalization by max(1, value)
    Relative error model means normalization by value

    Function is:  k0(x/10); Using relative error model, relative error should be good
        Error (absolute):                1.2e-14
        Error (relative):                2.0e-14
        Error (standard):                2.0e-15
        Error (standard, checks):        2.0e-15
        Error (standard, recompiled):    2.0e-15
        Scipy time (ms):                 511.0
        Build time (ms):                 189.3
        Approx time (ms):                38.2
        Approx time, with checks (ms):   34.0
        Approx time, recompiled  (ms):   26.0
        Points/Sec/Core, Millions:       96.2

    Function is:  k0(x/10); Using standard error model, relative error not guaranteed
        Error (absolute):                1.9e-13
        Error (relative):                3.5e+19
        Error (standard):                1.9e-13
        Error (standard, checks):        1.9e-13
        Error (standard, recompiled):    1.9e-13
        Scipy time (ms):                 508.4
        Build time (ms):                 52.9
        Approx time (ms):                19.6
        Approx time, with checks (ms):   29.9
        Approx time, recompiled  (ms):   14.9
        Points/Sec/Core, Millions:       167.3

    Function is:  y0(x); Using standard error model, relative error not guaranteed
        Error (absolute):                3.9e-15
        Error (relative):                9.2e-07
        Error (standard):                3.9e-15
        Error (standard, checks):        3.9e-15
        Error (standard, recompiled):    3.9e-15
        Scipy time (ms):                 286.7
        Build time (ms):                 499.9
        Approx time (ms):                48.8
        Approx time, with checks (ms):   61.3
        Approx time, recompiled  (ms):   45.0
        Points/Sec/Core, Millions:       55.5

    Function is:  hankel1(0, x); Using standard error model, relative error not guaranteed
        Error (absolute):                2.7e-15
        Error (relative):                7.2e-14
        Error (standard):                2.7e-15
        Error (standard, checks):        2.7e-15
        Error (standard, recompiled):    2.7e-15
        Scipy time (ms):                 2159.5
        Build time (ms):                 633.3
        Approx time (ms):                79.9
        Approx time, with checks (ms):   78.6
        Approx time, recompiled  (ms):   57.6
        Points/Sec/Core, Millions:       43.4

    Function is:  np.log(x); Using standard error model, relative error not guaranteed
        Error (absolute):                3.6e-15
        Error (relative):                7.5e-13
        Error (standard):                8.6e-16
        Error (standard, checks):        8.6e-16
        Error (standard, recompiled):    8.6e-16
        Scipy time (ms):                 15.1
        Build time (ms):                 51.3
        Approx time (ms):                27.4
        Approx time, with checks (ms):   33.6
        Approx time, recompiled  (ms):   15.9
        Points/Sec/Core, Millions:       156.9

    Function is:  1/x**8; Using relative error model, relative error should be good
        Error (absolute):                6.6e+64
        Error (relative):                1.6e-15
        Error (standard):                1.3e-15
        Error (standard, checks):        1.3e-15
        Error (standard, recompiled):    1.3e-15
        Scipy time (ms):                 191.8
        Build time (ms):                 273.6
        Approx time (ms):                26.0
        Approx time, with checks (ms):   36.1
        Approx time, recompiled  (ms):   21.4
        Points/Sec/Core, Millions:       116.9

    Function is:  1/sqrt(x) Using relative error model, relative error should be good
        Error (absolute):                1.5e-11
        Error (relative):                7.6e-16
        Error (standard):                7.6e-16
        Error (standard, checks):        7.6e-16
        Error (standard, recompiled):    7.6e-16
        Scipy time (ms):                 31.4
        Build time (ms):                 68.9
        Approx time (ms):                21.7
        Approx time, with checks (ms):   30.5
        Approx time, recompiled  (ms):   16.2
        Points/Sec/Core, Millions:       154.3

    Function is:  x*x; refinement test does not work; only gives one interval!
        Error (absolute):                4.7e-10
        Error (relative):                2.9e+09
        Error (standard):                4.4e-11
        Error (standard, checks):        4.4e-11
        Error (standard, recompiled):    4.4e-11
        Scipy time (ms):                 13.1
        Build time (ms):                 7.9
        Approx time (ms):                19.0
        Approx time, with checks (ms):   23.3
        Approx time, recompiled  (ms):   16.1
        Points/Sec/Core, Millions:       155.2

    Function is:  sin(1/x); should fail
Exception: Maximum number of intervals exceeded, try another function, increase 'mi', or increase 'n'.
```
