import numpy as np
import math
from QuantizeVector import vector_QIM_recover


def recovering_process(watermarked_signals: np.array, alpha: list[float], delta: list[int]):
    """
    Recovering
    :param watermarked_signals:
    :param alpha:
    :param delta:
    :return:
    """
    users = np.load('saveUsers.npy')
    dimension = users.shape[1]

    watermarked_signals = watermarked_signals.reshape(1, -1)
    length = math.floor(watermarked_signals.shape[1] / dimension)
    watermarked_signals = watermarked_signals[0:length * dimension].reshape(dimension, -1)

    for i in range(0, length):
        watermarked_signal = watermarked_signals[:, i].reshape((-1, 1))
        temp = sample_recovering(watermarked_signal, alpha, delta, users)
        watermarked_signals[:, i] = temp.reshape((1, -1))
        pass

    recovered_signals = watermarked_signals.reshape(-1)
    return recovered_signals


def sample_recovering(watermarked_signal: np.array, alpha: list[float], delta: list[int], users: np.array):
    """
    Single recovering
    :param watermarked_signal:
    :param alpha:
    :param delta:
    :param users:
    :return:
    """
    PW_vector = np.zeros(users.shape)  # Watermarked projection vector
    for i in range(PW_vector.shape[0]):
        PW_vector[i] = np.dot(watermarked_signal.reshape(1, -1), users[i]) / np.dot(users[i], users[i]) * users[i]

    PW = np.zeros(users.shape[0])  # Watermarked projection value
    for i in range(PW.shape[0]):
        PW[i] = np.dot(watermarked_signal.reshape(1, -1), users[i]) / np.dot(users[i], users[i])

    restore_P_vector = np.zeros(users.shape)  # Recovered projection vector
    for i in range(PW_vector.shape[0]):
        sw = PW_vector[i]
        restore_P_vector[i] = vector_QIM_recover(sw, alpha[i], delta[i])
        pass

    restore_P = np.zeros(users.shape[0])  # Recovered projection value
    for i in range(users.shape[0]):
        restore_P[i] = np.dot(restore_P_vector[i], users[i]) / np.dot(users[i], users[i])

    labda = np.diag([1 / np.sqrt(np.sum(np.power(users[j], 2))) for j in range(users.shape[0])])
    Ui = np.dot(np.dot(labda, users), users.T)
    K_ = np.dot(np.linalg.inv(Ui), restore_P - PW)
    restore_signal = watermarked_signal + np.dot(K_, users).reshape((-1, 1))
    return restore_signal


if __name__ == "__main__":
    Sw = np.load('watermarked_signals.npy')
    print(Sw)
    a = [0.8, 0.9]
    d = [3, 5]

    out = recovering_process(Sw, a, d)
    print(out)
