import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from math import atan2, degrees
from functools import partial

#data
m,n = 1000,2
X = np.random.randn(m,n)
T = np.random.randn(2,2)
X = np.matmul(X,T.T)   # same as: (T @ X.T).T


def draw_gaussian_ellipses(X, sp=None):
    sp = sp or plt.gca()

    #get gaussian parameters
    μ = X.mean(0)
    Σ = np.cov(X.T)
    E,λ,E_T = np.linalg.svd(Σ)
    
    #get angle, width, height
    x,y = E[:,0]
    θ = degrees(atan2(y,x))
    w,h = np.sqrt(λ)*2

    #draw points and gaussian ellipses
    sp.plot(*X.T, 'o', alpha=0.2)
    make_ellipse = partial(matplotlib.patches.Ellipse, xy=μ, angle=θ, color="#888888")
    ellipses = [make_ellipse(width=w*n, height=h*n, alpha=1/n) for n in range(1,4)]
    [sp.add_patch(e) for e in ellipses]
    return sp

sp = draw_gaussian_ellipses(X)
