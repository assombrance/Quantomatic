import numpy as np
import q_functions as qf

# matrix = np.matrix([[0, 5, 0, 2],
#                     [4, 0, 2, 0],
#                     [3, 0, 0, 17],
#                     [0, 0, 9, 0]])
# matrix = [[0, 5, 0, 2],
#           [4, 0, 2, 0],
#           [3, 0, 0, 17],
#           [0, 0, 9, 0]]
small_size = 31
small_matrix0 = np.matrix(np.random.randint(20, size=(small_size, small_size)))
small_matrix1 = np.matrix(np.random.randint(20, size=(small_size, small_size)))
size = small_size ** 2
matrix0 = np.matrix(np.random.randint(20, size=(size, size)))
matrix1 = np.matrix(np.random.randint(20, size=(size, size)))
iterations = 1


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


# def test_naive_multiplication():
#     result = naive_multiplication(matrix0, matrix1)
#     expected_result = matrix0 * matrix1
#     assert not (result - expected_result).any()


# def test_bench_naive():
#     for _ in np.arange(iterations):
#         naive_multiplication(matrix0, matrix1)
#     assert True


def test_bench_numpy():
    for _ in np.arange(iterations):
        _ = matrix0 * matrix1
    assert True


def test_bench_tensor():
    for _ in np.arange(iterations):
        _ = qf.tensor_product(small_matrix0, small_matrix1)
    assert True
