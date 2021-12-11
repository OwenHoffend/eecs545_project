import numpy as np
import matplotlib.pyplot as plt

def plot_WMD_dists():
    #Reproduce figure 6 of the WMD paper, plotting the distances of WCD, RWMD, and WMD (gensim)
    #For a single random test document (against all the training docs)
    train_split = 0.8
    num_train = int(train_split * 1000)

    ind = 0
    WMD = np.load("WMD_wmat_wiki.npy")[:, ind]
    RWMD = np.load("RWMD_wmat_wiki.npy")[:, ind]
    WCD = np.load("WCD_wmat_wiki.npy")[:, ind]
    wmd_argsort = np.argsort(WMD)

    print(WMD.shape)
    print(WCD.shape)
    print(RWMD.shape)

    plt.title("Distance vs Training Index")
    s = np.repeat(5, num_train)
    plt.scatter(range(num_train), WMD[wmd_argsort], s, label="WMD")
    plt.scatter(range(num_train), RWMD[wmd_argsort], s, label='RWMD')
    plt.scatter(range(num_train), WCD[wmd_argsort], s, label='WCD')
    plt.xlabel("Training Index")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

def main():
    plot_WMD_dists()

if __name__ == "__main__":
    main()