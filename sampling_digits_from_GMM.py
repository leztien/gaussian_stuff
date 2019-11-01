"""
demo on sampling digits from a GMM
"""


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
bunch = load_digits()
X = bunch.data
y = bunch.target

#PCA
from sklearn.decomposition import PCA
pca = PCA(0.99, whiten=True)
Xtr = pca.fit_transform(X)

#GMM
from sklearn.mixture import GaussianMixture
from functools import partial
n_components = range(80, 160, 10)
GMM = partial(GaussianMixture, covariance_type="full", n_init=3, verbose=3)
models = [GMM(n_components=n).fit(Xtr) for n in n_components]
aic = [md.aic(Xtr) for md in models]

ix = aic.index(min(aic))
md = models[ix]   # best_model
best_n_gaussians = n_components[ix]  # 110
b = md.converged_  #check to see whether the algorithm has converged

#sample
Xsample,ysample = md.sample(100)

#back PCA
Xsample_inverse = pca.inverse_transform(Xsample)

#visualize
plt.plot(n_components, aic)
fig, ax = plt.subplots(10,10, figsize=(4,4), subplot_kw={'xticks':[], 'yticks':[]})
fig.subplots_adjust(hspace=.05, wspace=.05)
[sp.imshow(Xsample_inverse[i].reshape(8,8), cmap='binary').set_clim(0,16) for i,sp in enumerate(ax.flat)]

