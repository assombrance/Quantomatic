# coding=UTF-8
"""
Test file
"""
from data import Node, Pi4Matrix, UsedFragment, Wire, Graph, Edge, ConnectionPoint, InterMatrixLink
import q_functions as qf
import numpy as np


def test_pi_4_matrix_node_to_matrix():
    node = Node('node')
    m = Pi4Matrix.node_to_matrix(node, 1, 1)
    expected_m = Pi4Matrix(np.matrix([[1, 0], [0, 1]]))
    assert m == expected_m


def test_fallback_node_to_matrix():
    node = Node('node')
    m = qf.fallback_node_to_matrix(node, 1, 1)
    expected_m = np.matrix([[1, 0], [0, 1]])
    assert not (m - expected_m).any()


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
    #
    # problem if an edge has the same name as a node

    a, b, c, d, e, f = Wire('a'), Wire('b'), Wire('c'), Wire('d'), Wire('e'), Wire('f')
    g, h = Wire('g'), Wire('h')
    n0, n1, n2, n3, n4, n5 = Node('n0'), Node('n1'), Node('n2'), Node('n3'), Node('n4'), Node('n5')
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13 = \
        Edge('e0', a, n0), Edge('e1', b, n1), Edge('e2', c, n2), \
        Edge('e3', n0, n3), Edge('e4', n0, n4), \
        Edge('e5', n1, n3), Edge('e6', n1, n4), Edge('e7', n1, n5), \
        Edge('e8', n2, n4), Edge('e9', n2, n5), \
        Edge('e10', d, n3), Edge('e11', e, n4), Edge('e12', f, n5), \
        Edge('e13', n0, n0)
    e14 = Edge('e14', g, h)
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


def test_split_and_reunite():
    b0 = Wire("b0")
    b1 = Wire("b1")
    b2 = Wire("b2")
    b3 = Wire("b3")
    v0 = Node("v0", 0.0, "X", 3)
    v1 = Node("v1", 0.0, "X", 3)
    v2 = Node("v2", 0.0, arity=3)
    v3 = Node("v3", 0.0, arity=3)
    e0 = Edge("e0", v0, v3)
    e1 = Edge("e1", v1, v2)
    e2 = Edge("e2", v2, v0)
    e3 = Edge("e3", v1, v3)
    e4 = Edge("e4", v0, b1)
    e5 = Edge("e5", v2, b0)
    e7 = Edge("e7", v1, b2)
    e8 = Edge("e8", v3, b3)
    inputs = [b1, b2]
    outputs = [b0, b3]
    nodes = [v3, v1, v0, v2]
    edges = [e3, e5, e0, e4, e2, e8, e1, e7]
    graph = Graph(nodes, edges, inputs, outputs)
    m1 = [[32, 0, 0, 0],
          [0, 0, 0, 32],
          [0, 0, 0, 32],
          [32, 0, 0, 0]]
    m2 = np.zeros((4, 4))
    m3 = np.zeros((4, 4))
    m4 = np.zeros((4, 4))
    expected_result = Pi4Matrix(m1, m2, m3, m4, 6)
    assert not (qf.split_and_reunite(graph) - expected_result).any()


def test_between_graphs_edges():
    b0 = Wire("b0")
    b1 = Wire("b1")
    b2 = Wire("b2")
    b3 = Wire("b3")
    v0 = Node("v0", 0.0, "X", 3)
    v1 = Node("v1", 0.0, "X", 3)
    v2 = Node("v2", 0.0, arity=3)
    v3 = Node("v3", 0.0, arity=3)
    e0 = Edge("e0", v0, v3)
    e1 = Edge("e1", v1, v2)
    e2 = Edge("e2", v2, v0)
    e3 = Edge("e3", v1, v3)
    e4 = Edge("e4", v0, b1)
    e5 = Edge("e5", v2, b0)
    e7 = Edge("e7", v1, b2)
    e8 = Edge("e8", v3, b3)
    inputs = [b1, b2]
    outputs = [b0, b3]
    nodes = [v3, v1, v0, v2]
    edges = [e3, e5, e0, e4, e2, e8, e1, e7]
    graph = Graph(nodes, edges, inputs, outputs)
    g1 = Graph([v0], [e4], [b1])
    g2 = Graph([v3, v1, v2], [e2, e0, e1, e3, e7, e8, e5], [b2], [b0, b3])
    result = qf.between_graphs_edges(g1, g2, graph)
    expected_result = [e2, e0]
    assert set(result) - set(expected_result) == set()


def test_no_node_edges_detection_true():
    assert qf.no_node_edges_detection([Edge("e0", Wire("b0"), Wire("b1"))])


def test_no_node_edges_detection_false():
    assert not qf.no_node_edges_detection([Edge("e0", Node("n0"), Wire("b1"))])


def test_no_node_matrix():
    #
    #   b4  b0  b5
    #     \  |  /
    #      --|-
    #        |
    #       b2
    #
    b0 = Wire("b0")
    b2 = Wire("b2")
    b4 = Wire("b4")
    b5 = Wire("b5")
    e0 = Edge("e0", b4, b5)
    e1 = Edge("e1", b0, b2)
    edges = [e0, e1]
    inputs = [b4, b0, b5]
    outputs = [b2]
    result = qf.no_node_matrix(edges, inputs, outputs)
    expected_result = Pi4Matrix([[1, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 1]])
    assert not (result - expected_result).any()


def test_wires_to_connection_point_node_sorted():
    we5 = Wire("e5")
    we8 = Wire("e8")
    v5 = Node("v5")
    v8 = Node("v8")
    v0 = Node("v0", node_type='hadamard', arity=2)
    v3 = Node("v3", node_type='hadamard', arity=2)
    e5 = Edge("e5", v5, v0)
    e6 = Edge("e6", v0, v3)
    e8 = Edge("e8", v8, v3)
    wires = [we5, we8]
    edges = [e8, e6, e5]
    nodes1 = [v0]
    nodes2 = [v3]
    result = qf.wires_to_connection_point_node_sorted(wires, edges, nodes1, nodes2, False)
    c0 = ConnectionPoint(is_matrix_2=False, is_out=False)
    c1 = ConnectionPoint(is_matrix_2=True, is_out=False)
    expected_result = [c0, c1]
    assert result == expected_result


def test_symmetric_difference():
    a = [1, 2, 3]
    b = [1, 3, 2]
    c = [1, 2]
    assert qf.symmetric_difference(a, b) == []
    assert qf.symmetric_difference(a, c) == [3]
    assert qf.symmetric_difference(c, a) == [3]


def test_fallback_tensor_power():
    h = UsedFragment([[1, 1], [1, -1]])
    expected_result = UsedFragment([[1, 1, 1, 1],
                                 [1, -1, 1, -1],
                                 [1, 1, -1, -1],
                                 [1, -1, -1, 1]])
    assert not (qf.tensor_power(h, 0) - [[1]]).any()
    assert not (qf.tensor_power(h, 1) - h).any()
    assert not (qf.tensor_power(h, 2) - expected_result).any()


def test_wires_to_connection_point_edge_sorted():
    b0 = Wire("b0")
    b1 = Wire("b1")
    b2 = Wire("b2")
    b3 = Wire("b3")
    e2 = Edge("e2", b3, b1)
    e1 = Edge("e1", b0, b2)
    wires = [b0, b1]
    edges1 = [e2]
    edges2 = [e1]
    result = qf.wires_to_connection_point_edge_sorted(wires, edges1, edges2, False)
    c0 = ConnectionPoint(is_matrix_2=True, is_out=False)
    c1 = ConnectionPoint(index=1, is_matrix_2=False, is_out=False)
    expected_result = [c0, c1]
    assert result == expected_result


def test_matrix_linker():
    v0 = Node("v0", arity=2, node_type="hadamard")
    v3 = Node("v3", arity=2, node_type="hadamard")
    v5 = Node("v5", angle=1.0, arity=3, node_type="X")
    we5 = Wire("e5")
    we6 = Wire("e6")
    e5 = Edge("e5", v5, v0)
    e6 = Edge("e6", v0, v3)
    graph1 = Graph([v0], [e5, e6], [we5], [we6])
    we8 = Wire("e8")
    v8 = Node("v8", 0.5, arity=4)
    e8 = Edge("e8", v8, v3)
    graph2 = Graph([v3], [e8, e6], [we8, we6], [])
    result = qf.matrix_linker(graph1, graph2)
    p1 = ConnectionPoint(False, True)
    p2 = ConnectionPoint(True, False, 1)
    expected_result = [InterMatrixLink(p1, p2)]
    assert result == expected_result


def test_filter_inputs_outputs_by_edges():
    b0 = Wire("b0")
    b1 = Wire("b1")
    b2 = Wire("b2")
    b3 = Wire("b3")
    b4 = Wire("b4")
    b5 = Wire("b5")
    e1 = Edge("e1", b0, b2)
    edges = [e1]
    inputs = [b4, b0, b5, b1]
    outputs = [b3, b2]
    result = qf.filter_inputs_outputs_by_edges(edges, inputs, outputs)
    expected_result = [b0], [b2]
    assert result == expected_result
