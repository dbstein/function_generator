import numpy as np

def standard_error_model(coefs):
    return np.abs(coefs[-2:]).max()/max(1, np.abs(coefs[0]))

def relative_error_model(coefs):
    return np.abs(coefs[-2:]).max()/np.abs(coefs[0])
