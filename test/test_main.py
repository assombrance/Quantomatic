from data import Node, Pi4Matrix, UsedFragment, Wire, Graph, Edge
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


def test_graph_addition():

    #  a  b  c
    #  |  |  |
    # n0 n1 n2
    #  |\/|\/|
    #  |/\|/\|
    # n3 n4 n5
    #  |  |  |
    #  d  e  f

    a, b, c, d, e, f = Wire('a'), Wire('b'), Wire('c'), Wire('d'), Wire('e'), Wire('f')
    n0, n1, n2, n3, n4, n5 = Node('0'), Node('1'), Node('2'), Node('3'), Node('4'), Node('5')
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12 = \
        Edge('0', a, n0), Edge('1', b, n1), Edge('2', c, n2), \
        Edge('3', n0, n3), Edge('4', n0, n4), \
        Edge('5', n1, n3), Edge('6', n1, n4), Edge('7', n1, n5), \
        Edge('8', n2, n4), Edge('9', n2, n5), \
        Edge('10', d, n3), Edge('11', e, n4), Edge('12', f, n5)
    inputs = [a, b, c]
    outputs = [d, e, f]
    nodes = [n0, n1, n2, n3, n4, n5]
    edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
    graph = Graph(nodes, edges, [], outputs)
    inputs_graph = Graph(inputs=inputs)
    assert graph + inputs_graph == Graph(nodes, edges, inputs, outputs)


def test_graph_subtraction():

    #  a  b  c
    #  |  |  |
    # n0 n1 n2
    #  |\/|\/|
    #  |/\|/\|
    # n3 n4 n5
    #  |  |  |
    #  d  e  f

    a, b, c, d, e, f = Wire('a'), Wire('b'), Wire('c'), Wire('d'), Wire('e'), Wire('f')
    n0, n1, n2, n3, n4, n5 = Node('0'), Node('1'), Node('2'), Node('3'), Node('4'), Node('5')
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12 = \
        Edge('0', a, n0), Edge('1', b, n1), Edge('2', c, n2), \
        Edge('3', n0, n3), Edge('4', n0, n4), \
        Edge('5', n1, n3), Edge('6', n1, n4), Edge('7', n1, n5), \
        Edge('8', n2, n4), Edge('9', n2, n5), \
        Edge('10', d, n3), Edge('11', e, n4), Edge('12', f, n5)
    inputs = [a, b, c]
    outputs = [d, e, f]
    nodes = [n0, n1, n2, n3, n4, n5]
    edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
    graph = Graph(nodes, edges, inputs, outputs)
    inputs_graph = Graph(inputs=inputs)
    assert graph - inputs_graph == Graph(nodes, edges, [], outputs)


def test_graph_augmentation():

    #  a  b  c
    #  |  |  |
    # n0 n1 n2
    #  |\/|\/|
    #  |/\|/\|
    # n3 n4 n5
    #  |  |  |
    #  d  e  f

    a, b, c, d, e, f = Wire('a'), Wire('b'), Wire('c'), Wire('d'), Wire('e'), Wire('f')
    n0, n1, n2, n3, n4, n5 = Node('0'), Node('1'), Node('2'), Node('3'), Node('4'), Node('5')
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13 = \
        Edge('0', a, n0), Edge('1', b, n1), Edge('2', c, n2), \
        Edge('3', n0, n3), Edge('4', n0, n4), \
        Edge('5', n1, n3), Edge('6', n1, n4), Edge('7', n1, n5), \
        Edge('8', n2, n4), Edge('9', n2, n5), \
        Edge('10', d, n3), Edge('11', e, n4), Edge('12', f, n5), \
        Edge('13', n0, n0)
    inputs = [a, b, c]
    outputs = [d, e, f]
    nodes = [n0, n1, n2, n3, n4, n5]
    edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13]
    graph = Graph(nodes, edges, inputs, outputs)
    inputs_graph = Graph(inputs=inputs, nodes=[n0], edges=[e0, e13])
    inputs_graph.augment(graph)
    expected_result = Graph([n0, n1, n2, n3, n4], [e0, e1, e2, e3, e4, e5, e6, e8, e13], inputs, [])
    assert inputs_graph == expected_result


def test_connected_graphs_split():

    #  a  b  c
    #  |  |  |
    # n0 n1 n2    g
    #  |\/|\/|    |
    #  |/\|/\|    h
    # n3 n4 n5
    #  |  |  |
    #  d  e  f

    a, b, c, d, e, f = Wire('a'), Wire('b'), Wire('c'), Wire('d'), Wire('e'), Wire('f')
    g, h = Wire('g'), Wire('h')
    n0, n1, n2, n3, n4, n5 = Node('0'), Node('1'), Node('2'), Node('3'), Node('4'), Node('5')
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13 = \
        Edge('0', a, n0), Edge('1', b, n1), Edge('2', c, n2), \
        Edge('3', n0, n3), Edge('4', n0, n4), \
        Edge('5', n1, n3), Edge('6', n1, n4), Edge('7', n1, n5), \
        Edge('8', n2, n4), Edge('9', n2, n5), \
        Edge('10', d, n3), Edge('11', e, n4), Edge('12', f, n5), \
        Edge('13', n0, n0)
    e14 = Edge('14', g, h)
    inputs = [a, b, c, g]
    outputs = [d, e, f, h]
    nodes = [n0, n1, n2, n3, n4, n5]
    edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14]
    graph = Graph(nodes, edges, inputs, outputs)
    small_graph = Graph(inputs=[g], outputs=[h], edges=[e14])
    g1, g2 = qf.connected_graphs_split(graph)
    assert g1 == small_graph or g2 == small_graph


def test_graph_bool_true():
    # assert g doesn't call bool(), hence the weird code with the if

    g = Graph([Node('n')])

    if g:
        assert True
    else:
        assert False


def test_graph_bool_false():

    g = Graph()

    assert not g
