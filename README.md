# function_generator: Fast, compilable recreation for slow functions

Updated with better lookup thanks to Robert Blackwell.  If your numba version is > 0.46.0, function inlining is used to provide significantly improved performance.

Results on my mac laptop:

```python
Testing function generator on 1000000 points
    minimum test value is: 1.00e-10
    maximum test value is: 9.99e+02

    Standard error model means normalization by max(1, value)
    Relative error model means normalization by value

    Function is:  k0(x); using standard error model, relative error not guaranteed
        Error (absolute):                2.5e-14
        Error (relative):                inf
        Error (standard):                9.3e-15
        Error (standard, checks):        9.3e-15
        Error (standard, recompiled):    9.3e-15
        Scipy time (ms):                 59.6
        Build time (ms):                 120.1
        Approx time (ms):                2.7
        Approx time, with checks (ms):   2.9
        Approx time, recompiled  (ms):   1.5
        Points/Sec/Core, Millions:       164.1

    Function is:  y0(x); using standard error model, relative error not guaranteed
        Error (absolute):                3.8e-15
        Error (relative):                1.4e-07
        Error (standard):                3.8e-15
        Error (standard, checks):        3.8e-15
        Error (standard, recompiled):    3.8e-15
        Scipy time (ms):                 29.7
        Build time (ms):                 520.9
        Approx time (ms):                6.3
        Approx time, with checks (ms):   5.0
        Approx time, recompiled  (ms):   5.0
        Points/Sec/Core, Millions:       50.2

    Function is:  hankel1(0, x); Using standard error model, relative error not guaranteed
        Error (absolute):                2.7e-15
        Error (relative):                7.2e-14
        Error (standard):                2.7e-15
        Error (standard, checks):        2.7e-15
        Error (standard, recompiled):    2.7e-15
        Scipy time (ms):                 228.0
        Build time (ms):                 635.5
        Approx time (ms):                8.5
        Approx time, with checks (ms):   7.9
        Approx time, recompiled  (ms):   5.6
        Points/Sec/Core, Millions:       44.9

    Function is:  np.log(x); Using standard error model, relative error not guaranteed
        Error (absolute):                3.6e-15
        Error (relative):                1.5e-13
        Error (standard):                7.1e-16
        Error (standard, checks):        7.1e-16
        Error (standard, recompiled):    7.1e-16
        Scipy time (ms):                 2.0
        Build time (ms):                 50.3
        Approx time (ms):                2.1
        Approx time, with checks (ms):   3.8
        Approx time, recompiled  (ms):   1.5
        Points/Sec/Core, Millions:       163.5

    Function is:  1/x**8; Using relative error model, relative error should be good
        Error (absolute):                6.6e+64
        Error (relative):                1.6e-15
        Error (standard):                9.6e-16
        Error (standard, checks):        9.6e-16
        Error (standard, recompiled):    9.6e-16
        Scipy time (ms):                 18.8
        Build time (ms):                 285.5
        Approx time (ms):                2.2
        Approx time, with checks (ms):   3.1
        Approx time, recompiled  (ms):   2.0
        Points/Sec/Core, Millions:       123.7
'''