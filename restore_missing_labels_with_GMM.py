"""

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from pandas import DataFrame


def restore_labels(ytrue, ypred):
    def func(df):
        mode = df["ytrue"].dropna().mode().values[0].astype("uint8")
        df["labels"] = mode
        return df
    df = DataFrame({"ytrue":ytrue, "ypred":ypred})
    labels = df.groupby(by="ypred").apply(func)["labels"].values
    return(labels)


#small example
n = None
yfull = [0,0,0,1,1,1,2,2,2,3,3,3,3,3]   # the target-array should look like this...
ytrue = [0,n,0,1,1,n,n,2,2,3,n,3,n,3]   # but some labels are missing
ypred = [1,1,1,0,0,0,3,3,3,3,2,2,2,2]   # this is what predicted targets may look like
labels = restore_labels(ytrue, ypred) #[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


#practical example of restoring labels with the help of GMM and my restore_labels function
X,y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0)
p = 0.3   # let 30% of the labels be missing
nx = np.random.choice(len(X), size=int(len(X)*p), replace=False)  # indeces of the missing values
ytrue = y.astype("float16") # must be float-type to assign nan-values
ytrue[nx] = np.nan          # corrupt the target-array by deleting 30% of labels


GMM = GaussianMixture(n_components=3, covariance_type="full", n_init=3)
GMM.fit(X)
ypred = GMM.predict(X)
labels = restore_labels(ytrue, ypred)   # restored labels
plt.scatter(*X.T, c=labels)
