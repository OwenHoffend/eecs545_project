import numpy as np
import matplotlib.pyplot as plt

def plot_WMD_dists():
    #Reproduce figure 6 of the WMD paper, plotting the distances of WCD, RWMD, and WMD (gensim)
    #For a single random test document (against all the training docs)
    train_split = 0.8
    num_train = int(train_split * 1000)

    RWMD = np.load("RWMD_wmat_old.npy")[:, 5] #First test example
    WCD = np.load("WCD_wmat_old.npy")[num_train+5, :num_train]
    rwmd_argsort = np.argsort(RWMD)

    print(WCD.shape)
    print(RWMD.shape)

    plt.title("Distance vs Training Index")
    s = np.repeat(5, num_train)
    plt.scatter(range(num_train), WCD[rwmd_argsort], s, label='WCD')
    plt.scatter(range(num_train), RWMD[rwmd_argsort], s, label='RWMD')
    plt.xlabel("Training Index")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

def main():
    plot_WMD_dists()

if __name__ == "__main__":
    main()