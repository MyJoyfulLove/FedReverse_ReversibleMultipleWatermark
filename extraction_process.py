import numpy as np
import math
from QuantizeVector import vector_QIM_extract


def extraction_process(watermarked_signals: np.array, delta: list[int]):
    """
    Extracting watermark
    :param watermarked_signals:
    :param delta: step size
    :return:
    """
    users = np.load('saveUsers.npy')
    dimension = users.shape[1]

    length = math.floor(len(watermarked_signals) / dimension)
    watermarked_signals = watermarked_signals[0:length * dimension].reshape(dimension, -1)

    extracted_messages = []

    for i in range(0, length):
        watermarked_signal = watermarked_signals[:, i]
        extracted_messages.append(sample_extraction(watermarked_signal, delta, users))
        pass
    return extracted_messages


def sample_extraction(watermarked_signal: np.array, delta: list[int], users: np.array):
    """
    Single extracting
    :param watermarked_signal:
    :param delta: step size
    :param users:
    :return:
    """
    PW_vector = np.zeros(users.shape)  # Projection vectors
    messages = []
    for i in range(PW_vector.shape[0]):
        PW_vector[i] = np.dot(watermarked_signal.reshape(1, -1), users[i]) / np.dot(users[i], users[i]) * users[i]
        messages.append(vector_QIM_extract(PW_vector[i], delta[i]))
    return messages


if __name__ == "__main__":
    Sw = np.load('watermarked_signals.npy')
    print(Sw)
    d = [3, 5]
    out = extraction_process(Sw, d)
    print(out)
