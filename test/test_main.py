from data import Node, Pi4Matrix, UsedFragment
import q_functions as qf
import numpy as np


def test_pi_4_matrix_node_to_matrix():
    node = Node('node')
    m = Pi4Matrix.node_to_matrix(node, 1, 1)
    expected_m = Pi4Matrix(np.matrix([[1, 0], [0, 1]]))
    assert m == expected_m


def test_fallback_tensor_product():
    m1 = UsedFragment(np.identity(2))
    m2 = UsedFragment([[0, 1],
                       [1, 0]])
    expected_result = UsedFragment(np.matrix([[0, 1, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 1],
                                              [0, 0, 1, 0]]))
    result = qf.tensor_product(m1, m2)
    assert not (result - expected_result).any()


def test_tensor_product():
    m1 = UsedFragment(np.identity(2))
    m2 = UsedFragment([[0, 1],
                       [1, 0]])
    expected_result = UsedFragment(np.matrix([[0, 1, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 1],
                                              [0, 0, 1, 0]]))
    result = m1.tensor_product(m2)
    assert not (result - expected_result).any()
