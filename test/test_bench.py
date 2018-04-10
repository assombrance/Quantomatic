import numpy as np

# matrix = np.matrix([[0, 5, 0, 2],
#                     [4, 0, 2, 0],
#                     [3, 0, 0, 17],
#                     [0, 0, 9, 0]])
# matrix = [[0, 5, 0, 2],
#           [4, 0, 2, 0],
#           [3, 0, 0, 17],
#           [0, 0, 9, 0]]
matrix0 = np.matrix(np.random.randint(20, size=(100, 6)))
matrix1 = np.matrix(np.random.randint(20, size=(6, 5)))


def naive_multiplication(m0: np.matrix, m1: np.matrix):
    shape_m0 = m0.shape
    shape_m1 = m1.shape
    m01 = np.zeros((shape_m0[0], shape_m1[1]))
    if shape_m0[1] != shape_m1[0]:
        raise NameError("Dimensions not aligned : dim0(m0)=" + str(shape_m0[1]) + " and dim1(m1)=" + str(shape_m1[0]))
    for i in np.arange(shape_m0[0]):
        for j in np.arange(shape_m1[1]):
            for k in np.arange(shape_m1[0]):
                m01[i][j] += m0.item((i, k))*m1.item((k, j))
    return m01


def test_naive_multiplication():
    result = naive_multiplication(matrix0, matrix1)
    expected_result = matrix0 * matrix1
    # print(result)
    # print(expected_result)
    assert not (result - expected_result).any()


def test_bench_naive():
    for _ in np.arange(1000):
        naive_multiplication(matrix0, matrix1)
    assert True


def test_bench_numpy():
    for _ in np.arange(1000):
        _ = matrix0 * matrix1
    assert True
