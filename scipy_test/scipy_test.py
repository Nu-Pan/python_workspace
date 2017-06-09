import numpy as np
from scipy import integrate

def computePi(x):
    return 4/(1+x**2)

print (integrate.quad(computePi, 0, 1))
