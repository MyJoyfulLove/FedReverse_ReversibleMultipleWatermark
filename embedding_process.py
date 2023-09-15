import numpy as np
import math
from TheQR import GivenRot
from QuantizeVector import vector_QIM_embed, is_collinear


def create_users(user_num: int, n: int) -> np.array:
    """
    create non-vertical user vectors
    :param user_num: number of users
    :param n: dimension of vector (space)
    :return: users matrix
    """
    if user_num > n:
        print("Number of users exceeds dimension")
        quit()

    users = np.random.randint(1, 5, (user_num, n))
    while True:
        temp = False
        if user_num == 1:
            return users
        for i in range(user_num - 2):
            for j in range(i + 1, user_num - 1):
                if is_collinear(users[i], users[j]):
                    temp = True
                    break
        if temp:
            users = np.random.randint(1, 100, (user_num, n))
        else:
            return users


def create_users2(user_num: int, n: int) -> np.array:
    """
    create vertical user vectors
    :param user_num: number of users
    :param n: dimension of vector (space)
    :return: users
    """
    if user_num > n:
        print("Number of users exceeds dimension")
        quit()
    users, _ = GivenRot(np.random.randint(1, 100, (n, n)))
    return users[:user_num]


def embedding_process(signals: np.array, messages: list, alpha: list[float], delta: list[int], dimension: int, user_num: int):
    """
    Embedding
    :param signals: host signals
    :param messages:
    :param alpha: scaling factor
    :param delta: step size
    :param dimension:
    :param user_num:
    :return:
    """
    if signals.shape[0] % (dimension * user_num) != 0:
        print("The number of signals needs to be divisible (dimension × number of users)")
        quit()
    if len(messages) < signals.shape[0] / dimension * user_num:
        print("Not enough messages")
        quit()

    users = create_users2(user_num, dimension)
    np.save('saveUsers', users)
    with open('saveUsers.txt', 'w') as f:
        for row in users:
            for x in row:
                f.write(str(x) + ",")
            f.write("\n")

    length = math.floor(signals.shape[0] / dimension)
    signals_temp = signals[0:length * dimension].reshape(dimension, -1)

    for i in range(0, length):
        signal = signals_temp[:, i].reshape((signals_temp.shape[0], -1))
        message = messages[i * user_num: i * user_num + user_num]
        temp = sample_embedding(signal, message, alpha, delta, users)
        signals_temp[:, i] = temp.reshape((1, -1))
        pass

    watermarked_signals = signals_temp.reshape(-1)
    return watermarked_signals


def sample_embedding(signal: np.array, message: np.array, alpha: list[float], delta: list[int], users: np.array):
    """
    Single embedding
    :param signal: single host signal
    :param message:
    :param alpha: scaling factor
    :param delta: step size
    :param users:
    :return:
    """
    P_vector = np.zeros(users.shape)  # Projection vector
    for i in range(P_vector.shape[0]):
        P_vector[i] = np.dot(signal.reshape(1, -1), users[i]) / np.dot(users[i], users[i]) * users[i]

    P = np.zeros(users.shape[0])  # Projection value
    for i in range(P.shape[0]):
        P[i] = np.dot(signal.reshape(1, -1), users[i]) / np.sqrt(np.dot(users[i], users[i]))

    # Project the original signal to each user vector and embed it on the projected vector
    WP_vector = np.zeros(users.shape)  # Watermarked projection vector
    for i in range(P_vector.shape[0]):
        pw = P_vector[i]
        WP_vector[i] = vector_QIM_embed(pw, message[i], alpha[i], delta[i])
        pass

    WP = np.zeros(users.shape[0])  # Watermarked projection value
    for i in range(users.shape[0]):
        WP[i] = np.dot(WP_vector[i], users[i]) / np.dot(users[i], users[i])

    # Merge the embedded projection vectors back into a watermark vector
    labda = np.diag([1 / np.sqrt(np.sum(np.power(users[j], 2))) for j in range(users.shape[0])])
    Ui = np.dot(np.dot(labda, users), users.T)
    K = np.dot(np.linalg.inv(Ui), WP - P)
    watermarked_signal = signal + np.dot(K, users).reshape((-1, 1))
    return watermarked_signal


if __name__ == "__main__":
    S = np.array([8.,  2, -2, 5, 6, 7.4, 8.,  2.5, 2, -5, 6, 7])
    M = [1, 0, 1, 1, 0, 1, 0, 0]
    a = [0.8, 0.9]
    d = [3, 5]
    dim = 3
    un = 2
    out = embedding_process(S, M, a, d, dim, un)
    print(out)
    np.save('watermarked_signals', out)
