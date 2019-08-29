# function_generator: Fast, compilable recreation for slow functions

Results on my mac laptop:

```python
Testing function generator on 10000000 points
    minimum test value is: 1.00e-10
    maximum test value is: 9.99e+02

    Standard error model means normalization by max(1, value)
    Relative error model means normalization by value

    Function is:  y0(x); using standard error model, relative error not guaranteed
        Error (absolute):                4.4e-15
        Error (relative):                1.0e-06
        Error (standard):                4.4e-15
        Error (standard, checks):        4.4e-15
        Error (standard, recompiled):    4.4e-15
        Scipy time (ms):                 350.6
        Build time (ms):                 104.5
        Approx time (ms):                114.6
        Approx time, with checks (ms):   138.3
        Approx time, recompiled  (ms):   129.3
        Points/Sec/Core, Millions:       21.8

    Function is:  hankel1(0, x); Using standard error model, relative error not guaranteed
        Error (absolute):                3.6e-15
        Error (relative):                6.0e-14
        Error (standard):                2.6e-15
        Error (standard, checks):        2.6e-15
        Error (standard, recompiled):    2.6e-15
        Scipy time (ms):                 2123.6
        Build time (ms):                 117.4
        Approx time (ms):                139.8
        Approx time, with checks (ms):   150.8
        Approx time, recompiled  (ms):   128.9
        Points/Sec/Core, Millions:       17.9

    Function is:  np.log(x); Using standard error model, relative error not guaranteed
        Error (absolute):                6.2e-15
        Error (relative):                1.6e-12
        Error (standard):                3.8e-15
        Error (standard, checks):        3.8e-15
        Error (standard, recompiled):    3.8e-15
        Scipy time (ms):                 15.3
        Build time (ms):                 17.8
        Approx time (ms):                62.2
        Approx time, with checks (ms):   95.6
        Approx time, recompiled  (ms):   79.8
        Points/Sec/Core, Millions:       40.2

    Function is:  1/x**8; Using relative error model, relative error should be good
        Error (absolute):                7.9e+64
        Error (relative):                2.8e-15
        Error (standard):                2.4e-15
        Error (standard, checks):        2.4e-15
        Error (standard, recompiled):    2.4e-15
        Scipy time (ms):                 198.9
        Build time (ms):                 45.5
        Approx time (ms):                81.2
        Approx time, with checks (ms):   101.4
        Approx time, recompiled  (ms):   93.2
        Points/Sec/Core, Millions:       30.8

'''