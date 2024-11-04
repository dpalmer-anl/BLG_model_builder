import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    csfont = {'fontname':'serif',"size":13}
    nsplits = np.array([2,3,4,5,6])
    n_train_interlayer = 44 - (44/nsplits)
    n_train_intralayer = 50 - (50/nsplits)

    classical_intralayer_test_loss = np.zeros(len(nsplits))
    tetb_intralayer_test_loss = np.zeros(len(nsplits))
    for i,n in enumerate(nsplits):
        classical_data = np.load("ensembles/Classical_energy_intralayer_Kfold_n_"+str(n)+"_ensemble.npz")
        test_loss = classical_data["test_loss"]
        classical_intralayer_test_loss[i] = np.mean(test_loss[test_loss<3])

        tetb_data = np.load("ensembles/TETB_intralayer_Kfold_n_"+str(n)+"_ensemble.npz")
        test_loss = tetb_data["test_loss"]
        tetb_intralayer_test_loss[i] = np.mean(test_loss[test_loss<3])

    plt.plot(n_train_intralayer,classical_intralayer_test_loss,label = "Classical",marker="*",c="black",linestyle="dashed")
    plt.plot(n_train_intralayer,tetb_intralayer_test_loss,label = r"TETB($n_{kp}$ = 121)",marker="s",c="blue",linestyle="dotted")
    plt.xlabel("Number of Training Data Points",**csfont)
    plt.ylabel("CV Score",**csfont)
    plt.legend()
    plt.savefig("figures/kfold_cv_intralayer.png")
    plt.clf()

    classical_interlayer_test_loss = np.zeros(len(nsplits))
    tetb_interlayer_test_loss = np.zeros(len(nsplits))
    for i,n in enumerate(nsplits):
        classical_data = np.load("ensembles/Classical_energy_interlayer_Kfold_n_"+str(n)+"_ensemble.npz")
        test_loss = classical_data["test_loss"]
        classical_interlayer_test_loss[i] = np.mean(test_loss[test_loss<3])

        tetb_data = np.load("ensembles/TETB_interlayer_Kfold_n_"+str(n)+"_ensemble.npz")
        test_loss = tetb_data["test_loss"]
        tetb_interlayer_test_loss[i] = np.mean(test_loss[test_loss<3])

    plt.plot(n_train_interlayer,classical_interlayer_test_loss,label = "Classical",marker="*",c="black",linestyle="dashed")
    plt.plot(n_train_interlayer,tetb_interlayer_test_loss,label = r"TETB($n_{kp}$ = 121)",marker="s",c="blue",linestyle="dotted")
    plt.xlabel("Number of Training Data Points",**csfont)
    plt.ylabel("CV Score",**csfont)
    plt.legend()
    plt.savefig("figures/kfold_cv_interlayer.png")
    plt.clf()


