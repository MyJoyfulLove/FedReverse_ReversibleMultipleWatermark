import numpy as np


def is_collinear(vec1, vec2):  # Determine whether the vectors are collinear, True means collinear
    cosine = np.dot(vec1, vec2) / np.sqrt(np.sum(np.power(vec1, 2))) / np.sqrt(np.sum(np.power(vec2, 2)))
    return True if 1 - 1e-6 < abs(cosine) < 1 + 1e-6 else False


def vector_QIM_embed(signal: np.array, message: int, alpha: float, delta: int):
    """
    Only change vector length to embed information
    :param signal:
    :param message: {0,1}
    :param alpha: scaling factor ∈[(r-1)/r,1]，r is number of elements in message set，here is 2
    :param delta: step size
    :return:
    """
    unit_vector = signal / np.sqrt(np.dot(signal, signal)) * delta  # 属于一个消息的单位向量
    if not is_collinear(unit_vector, signal):
        print("non-collinear")
        quit()
        pass
    short = np.sqrt(np.sum(np.power(signal, 2))) // np.sqrt(np.sum(np.power(unit_vector, 2)))
    if message == 0:
        quantized_vector = unit_vector * (short if short % 2 == 0 else short + 1)
        pass
    else:  # message == 1
        quantized_vector = unit_vector * (short if short % 2 == 1 else short + 1)
        pass
    watermarked_vector = alpha * quantized_vector + (1 - alpha) * signal
    return watermarked_vector


def vector_QIM_extract(watermarked_signal: np.array, delta: int):
    """
    Extracting
    :param watermarked_signal:
    :param delta: step size
    :return:
    """
    unit_vector = watermarked_signal / np.sqrt(np.dot(watermarked_signal, watermarked_signal)) * delta
    length = np.sqrt(np.sum(np.power(watermarked_signal, 2))) / np.sqrt(np.sum(np.power(unit_vector, 2)))
    extract_message = round(length) % 2
    return extract_message


def vector_QIM_recover(watermarked_signal: np.array, alpha: float, delta: int):
    """
    Recovering
    :param watermarked_signal:
    :param alpha: scaling factor
    :param delta: step size
    :return:
    """
    unit_vector = watermarked_signal / np.sqrt(np.dot(watermarked_signal, watermarked_signal)) * delta
    length = np.sqrt(np.sum(np.power(watermarked_signal, 2))) / np.sqrt(np.sum(np.power(unit_vector, 2)))
    quantized_vector = unit_vector * round(length)
    recovered_signal = (watermarked_signal - alpha * quantized_vector) / (1 - alpha)
    return recovered_signal


if __name__ == "__main__":
    s = np.array([0.8, 0])
    print("Original signal")
    print(s)
    m = 1
    print("Message")
    print(m)

    d = 5
    a = 0.8

    print("Watermarked signal")
    sw = vector_QIM_embed(s, m, a, d)
    print(sw)
    print("Extracted message")
    print(vector_QIM_extract(sw, d))
    print("Recovered signal")
    print(vector_QIM_recover(sw, a, d))
