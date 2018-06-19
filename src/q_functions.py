# coding=UTF-8
"""
Module focused on going through the graph.
"""
from copy import deepcopy
from typing import List
from collections import Iterable
import numpy as np

import divide_conquer

from data import UsedFragment, Node, Wire, Edge, GenericMatrix, InterMatrixLink, ConnectionPoint, Graph


def split_and_reunite(graph: Graph) -> GenericMatrix:
    """split_and_reunite(nodes: List[Node], edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> GenericMatrix

    Recursive function taking in a graph and returning the corresponding matrix.

    To do so, split the graph in two, passes the two halves to it's next iteration and reunite the two matrix obtained
    using the :ref:`fusion_matrices <fusion_matrices>` method from :ref:`divide_conquer`.

    The main part of this function is converting the graph format to the matrix format. tmp

    Args:
        graph (Graph): diagram considered

    Returns:
        GenericMatrix: matrix corresponding to the given diagram
    """
    if len(graph.nodes) == 0:
        return no_node_matrix(graph.edges, graph.inputs, graph.outputs)
    elif len(graph.nodes) == 1 and not no_node_edges_detection(graph.edges):
        try:
            return UsedFragment.node_to_matrix(graph.nodes[0], len(graph.inputs), len(graph.outputs))
        except AttributeError:
            return fallback_node_to_matrix(graph.nodes[0], len(graph.inputs), len(graph.outputs))
    else:
        graph1, graph2 = connected_graphs_split(graph)
        if not graph2:
            if no_node_edges_detection(graph.edges):
                # degenerate cases, when a graph contains only wires
                first_half_nodes = []
                second_half_nodes = graph.nodes

                first_half_edges, first_half_inputs, first_half_outputs = \
                    filter_edges_inputs_outputs_by_nodes_negative(second_half_nodes, graph.edges,
                                                                  graph.inputs, graph.outputs)
            else:
                half = len(graph.nodes) // 2
                first_half_nodes = graph.nodes[:half]
                second_half_nodes = graph.nodes[half:]

                first_half_edges, first_half_inputs, first_half_outputs = \
                    filter_edges_inputs_outputs_by_nodes(first_half_nodes, graph.edges, graph.inputs, graph.outputs)

            second_half_edges, second_half_inputs, second_half_outputs = \
                filter_edges_inputs_outputs_by_nodes(second_half_nodes, graph.edges, graph.inputs, graph.outputs)

            first_half_matrix = split_and_reunite(Graph(first_half_nodes, first_half_edges,
                                                        first_half_inputs, first_half_outputs))
            second_half_matrix = split_and_reunite(Graph(second_half_nodes, second_half_edges,
                                                         second_half_inputs, second_half_outputs))

            inter_matrix_link = matrix_linker(first_half_outputs, second_half_outputs)
        else:
            first_half_nodes = graph1.nodes
            second_half_nodes = graph2.nodes
            first_half_matrix = split_and_reunite(graph1)
            second_half_matrix = split_and_reunite(graph2)
            inter_matrix_link = []

        input_connections = wires_to_connection_point_node_sorted(graph.inputs, graph.edges,
                                                                  first_half_nodes, second_half_nodes, False)
        output_connections = wires_to_connection_point_node_sorted(graph.outputs, graph.edges,
                                                                   first_half_nodes, second_half_nodes, True)

        return divide_conquer.fusion_matrices(first_half_matrix, second_half_matrix, input_connections,
                                              output_connections, inter_matrix_link)


def connected_graphs_split(graph: Graph) -> (Graph, Graph):
    """ If possible, splits a graph in two separate graphs, the first one has to be connected.

    Args:
        graph (Graph): graph to be split in two

    Returns:
        (Graph, Graph): a connected graph and the rest of the edges, nodes and so on ...
    """
    # possible improvement, return balanced graphs
    if graph.nodes:
        increasing_graph = Graph(nodes=[graph.nodes[0]])
    elif graph.inputs:
        increasing_graph = Graph(inputs=[graph.inputs[0]])
    elif graph.outputs:
        increasing_graph = Graph(outputs=[graph.outputs[0]])
    else:
        # graph empty
        return Graph([], [], [], []), deepcopy(graph)
    while increasing_graph.augment(graph):
        pass
    leftover = graph - increasing_graph
    return increasing_graph, leftover


def no_node_edges_detection(edges: List[Edge]) -> bool:
    """no_node_edges_detection(edges: List[Edge]) -> bool

    Used to check if the function no_node_matrix has to be used, return *True* if at least one edge doesn't contain
    a node, *False* otherwise.

    Args:
        edges (List[Edge]): list of edges to be checked

    Returns:
        bool: is there one edge without a node in it ?
    """
    for edge in edges:
        if not isinstance(edge.n1, Node) and not isinstance(edge.n2, Node):
            return True
    return False


def no_node_matrix(edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> GenericMatrix:
    """no_node_matrix(edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> GenericMatrix

    Works similarly to split and reunite but without any node

    Args:
        edges (List[Edge]): edges in the diagram considered
        inputs (List[Wire]): inputs in the diagram considered
        outputs (List[Wire]): outputs in the diagram considered

    Returns:
        GenericMatrix: matrix corresponding to the given diagram
    """
    if 2*len(edges) != len(inputs) + len(outputs):
        raise ValueError("len(edges) != len(inputs) + len(outputs) : len(edges) == %d, len(inputs) == %d and "
                         "len(outputs) == %d" % (len(edges), len(inputs), len(outputs)))
    if len(edges) == 0:
        return UsedFragment(1)
    if len(edges) == 1:
        if len(inputs) == 0:
            return UsedFragment([[1],
                                 [0],
                                 [0],
                                 [1]])
        if len(inputs) == 1:
            return UsedFragment([[1, 0],
                                 [0, 1]])
        if len(inputs) == 2:
            return UsedFragment([[1, 0, 0, 1]])
        else:
            raise RuntimeError("Unhandled case of no node matrix")
    else:
        half = len(edges) // 2
        first_half_edges = edges[:half]
        second_half_edges = edges[half:]

        first_half_inputs, first_half_outputs = filter_inputs_outputs_by_edges(first_half_edges, inputs, outputs)
        second_half_inputs, second_half_outputs = filter_inputs_outputs_by_edges(second_half_edges, inputs, outputs)

        first_half_matrix = no_node_matrix(first_half_edges, first_half_inputs, first_half_outputs)
        second_half_matrix = no_node_matrix(second_half_edges, second_half_inputs, second_half_outputs)

        input_connections = wires_to_connection_point_edge_sorted(inputs, first_half_edges, second_half_edges, False)
        output_connections = wires_to_connection_point_edge_sorted(outputs, first_half_edges, second_half_edges, True)

        return divide_conquer.fusion_matrices(first_half_matrix, second_half_matrix, input_connections,
                                              output_connections, [])


def wires_to_connection_point_node_sorted(wires: List[Wire], edges: List[Edge], nodes_group_1: List[Node],
                                          nodes_group_2: List[Node], is_output: bool) -> List[ConnectionPoint]:
    """wires_to_connection_point_node_sorted(wires: List[Wire], edges: List[Edge], nodes_group_1: List[Node], nodes_group_2: List[Node], is_output: bool) -> List[ConnectionPoint]

    Part of the conversion system from the data under the graph form (nodes, wires and edges) to the matrix form
    (links, connection points). In particular, Does the *ConnectionPoint* part.

    Args:
        wires (List[Wire]): list of the considered wires in the diagram
        edges (List[Edge]: list of all the edges in the diagram
        nodes_group_1 (List[Node]): list of the first half of the nodes in the diagram
        nodes_group_2 (List[Node]): list of the first half of the nodes in the diagram
        is_output (bool): are the wires given outputs or inputs

    Returns:
        List[ConnectionPoint]: connection points to this graph
    """
    connection_points = []
    connection_points_dict = {}

    def _wires_to_connection_point_node_shortcut(_wires, _edges, _nodes_group, _is_output, _is_matrix_2, _cp_dict):
        index = 0
        for node in _nodes_group:
            # we are iterating over the nodes to ensure an order, indeed since the two halves are done depending on the
            # node order in this V2.0 of the algorithm, we can simply ensure the order by giving the I/O indexes
            # depending on node position
            for edge in edges:
                if node in edge:
                    wire_index = None
                    if set(edge).intersection(wires):
                        # in this case, the edge links a node to an original I/O
                        wire_index = wires.index(list(set(edge).intersection(wires))[0])
                    if edge.name in [wire.name for wire in wires]:
                        # when a diagram is split, the edges between two diagrams are considered as connected to a new
                        # virtual output, with the name of the edge as wire name
                        wire_index = wires.index(Wire(edge.name))

                    if wire_index is not None:
                        point = ConnectionPoint(is_matrix_2=_is_matrix_2, is_out=is_output, index=index)
                        _cp_dict[wire_index] = point
                        index += 1

        return _cp_dict

    def _wires_to_connection_point_no_node_shortcut(_wires, _edges, _nodes_group_2, _is_output, _is_matrix_2, _cp_dict):
        # adds the wires connected to none of the nodes from _node_group_2
        index = 0
        for wire in _wires:
            unlinked = True
            for _node in _nodes_group_2:
                for _edge in _edges:
                    if _node in _edge and wire in _edge:
                        unlinked = False
            if unlinked:
                _cp_dict[index] = ConnectionPoint(is_matrix_2=_is_matrix_2, is_out=is_output, index=index)
            index += 1

        return _cp_dict

    # this test is only done for the first group since for this algorithm, only the first group may be empty
    if nodes_group_1:
        connection_points_dict = _wires_to_connection_point_node_shortcut(wires, edges, nodes_group_1, is_output,
                                                                          False, connection_points_dict)
    else:
        connection_points_dict = _wires_to_connection_point_no_node_shortcut(wires, edges, nodes_group_2, is_output,
                                                                             False, connection_points_dict)
    if nodes_group_2:
        connection_points_dict = _wires_to_connection_point_node_shortcut(wires, edges, nodes_group_2, is_output,
                                                                          True, connection_points_dict)
    else:
        connection_points_dict = _wires_to_connection_point_no_node_shortcut(wires, edges, nodes_group_1, is_output,
                                                                             True, connection_points_dict)
    for i in np.arange(len(wires)):
        connection_points.append(connection_points_dict[i])
    return connection_points


def wires_to_connection_point_edge_sorted(wires: List[Wire], edges_group_1: List[Edge], edges_group_2: List[Edge],
                                          is_output: bool) -> List[ConnectionPoint]:
    """wires_to_connection_point_edge_sorted(wires: List[Wire], edges_group_1: List[Edge], edges_group_2: List[Edge], is_output: bool) -> List[ConnectionPoint]

    Part of the conversion system from the data under the graph form (wires and edges) to the matrix form
    (links, connection points). In particular, Does the *ConnectionPoint* part (specialised for graphs without nodes).

    Args:
        wires (List[Wire]): list of the considered wires in the diagram
        edges_group_1 (List[Edge]: list of all the edges in the diagram
        edges_group_2 (List[Edge]: list of all the edges in the diagram
        is_output (bool): are the wires given outputs or inputs

    Returns:
        List[ConnectionPoint]: connection points to this graph
    """
    connection_points = []
    edges_group_1_wire_set = {edge.n1 for edge in edges_group_1} | {edge.n2 for edge in edges_group_1}
    edges_group_2_wire_set = {edge.n1 for edge in edges_group_2} | {edge.n2 for edge in edges_group_2}

    index = 0
    for wire in wires:
        if wire in edges_group_1_wire_set:
            connection_points.append(ConnectionPoint(False, is_output, index))
        if wire in edges_group_2_wire_set:
            connection_points.append(ConnectionPoint(True, is_output, index))
        index += 1

    return connection_points


def matrix_linker(m1_outputs: List[Wire], m2_outputs: List[Wire]) -> List[InterMatrixLink]:
    """matrix_linker(m1_outputs: List[Wire], m2_outputs: List[Wire]) -> List[InterMatrixLink]

    Creates a list of *InterMatrixLink* from the common outputs of **m1** and **m2**.

    Links between the two matrices are forced to be between their outputs, that's why you don't need the inputs.

    Args:
        m1_outputs (List[Wire]): outputs from **m1** as a *Wire* list
        m2_outputs (List[Wire]): outputs from **m2** as a *Wire* list

    Returns:
        List[InterMatrixLink]: links between **m1** and **m2** as a *InterMatrixLink* list
    """
    inter_matrix_link = []
    for m1_output in m1_outputs:
        # because of the way the filter has been done, not inter matrix link should be between anything else than
        # the outputs of the first matrix and the second one
        if m1_output in m2_outputs:
            index_n1 = m1_outputs.index(m1_output)
            index_n2 = m2_outputs.index(m1_output)
            connection_point1 = ConnectionPoint(is_matrix_2=False, is_out=True, index=index_n1)
            connection_point2 = ConnectionPoint(is_matrix_2=True, is_out=True, index=index_n2)
            link = InterMatrixLink(connection_point1, connection_point2)
            inter_matrix_link.append(link)
    return inter_matrix_link


def filter_edges_inputs_outputs_by_nodes(nodes: List[Node], edges: List[Edge], inputs: List[Wire],
                                         outputs: List[Wire]) -> (List[Edge], List[Wire], List[Wire]):
    """filter_edges_inputs_outputs_by_nodes(nodes: List[Node], edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> List[Edge], List[Wire], List[Wire]

    Since the node list is split in two, many edges and wires don't need to be considered for the next iteration of
    each half. Thus, the edges, inputs and outputs are filtered so they are not passed to the next iteration if they are
    not in relation with the given nodes

    Args:
        nodes (List[Node]): nodes to apply the filter from
        edges (List[Edge]): edges to be filtered
        inputs (List[Wire]): inputs to be filtered
        outputs (List[Wire]): outputs to be filtered

    Returns:
        List[Edge], List[Wire], List[Wire]: edges, inputs and outputs without the members not in relation with the given
        nodes
    """
    new_edges = edges[:]
    new_inputs = inputs[:]
    new_outputs = outputs[:]
    for edge in edges:
        if not set(edge).intersection(nodes):  # edge doesn't contain any node from the list
            new_edges.remove(edge)
            if set(edge).intersection(inputs):
                new_inputs.remove(list(set(edge).intersection(inputs))[0])
            if Wire(edge.name) in inputs:
                new_inputs.remove(Wire(edge.name))
            if set(edge).intersection(outputs):
                new_outputs.remove(list(set(edge).intersection(outputs))[0])
            if Wire(edge.name) in outputs:
                new_outputs.remove(Wire(edge.name))
        elif len(set(edge).intersection(nodes)) == 1 and Wire(edge.name) not in new_outputs and \
                not set(edge).intersection(inputs) and not set(edge).intersection(outputs):
            new_wire = Wire(edge.name)
            new_outputs.append(new_wire)
    return new_edges, new_inputs, new_outputs


def filter_edges_inputs_outputs_by_nodes_negative(nodes: List[Node], edges: List[Edge], inputs: List[Wire],
                                                  outputs: List[Wire]) -> (List[Edge], List[Wire], List[Wire]):
    """filter_edges_inputs_outputs_by_nodes_negative(nodes: List[Node], edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> List[Edge], List[Wire], List[Wire]

    Since the node list is split in two, many edges and wires don't need to be considered for the next iteration of
    each half. Thus, the edges, inputs and outputs are filtered so they are not passed to the next iteration if they are
    in relation with the given nodes from the other half

    Args:
        nodes (List[Node]): nodes to apply the filter from
        edges (List[Edge]): edges to be filtered
        inputs (List[Wire]): inputs to be filtered
        outputs (List[Wire]): outputs to be filtered

    Returns:
        List[Edge], List[Wire], List[Wire]: edges, inputs and outputs without the members in relation with the given
        nodes
    """
    new_edges = edges[:]
    new_inputs = inputs[:]
    new_outputs = outputs[:]
    for edge in edges:
        if set(edge).intersection(nodes):  # edge doesn't contain any node from the list
            new_edges.remove(edge)
            if set(edge).intersection(inputs):
                new_inputs.remove(list(set(edge).intersection(inputs))[0])
            if Wire(edge.name) in inputs:
                new_inputs.remove(Wire(edge.name))
            if set(edge).intersection(outputs):
                new_outputs.remove(list(set(edge).intersection(outputs))[0])
            if Wire(edge.name) in outputs:
                new_outputs.remove(Wire(edge.name))
    return new_edges, new_inputs, new_outputs


def filter_inputs_outputs_by_edges(edges: List[Edge],
                                   inputs: List[Wire], outputs: List[Wire]) -> (List[Wire], List[Wire]):
    """filter_inputs_outputs_by_edges(edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> List[Wire], List[Wire]

    Since the edge list is split in two, many wires don't need to be considered for the next iteration of
    each half. Thus, the inputs and outputs are filtered so they are not passed to the next iteration if they are
    not in relation with the given edges

    Args:
        edges (List[Edge]): edges to apply the filter from
        inputs (List[Wire]): inputs to be filtered
        outputs (List[Wire]): outputs to be filtered

    Returns:
        List[Wire], List[Wire]: inputs and outputs without the members not in relation with the given edges
    """
    new_inputs = inputs[:]
    new_outputs = outputs[:]
    for input_wire in inputs:
        for edge in edges:
            if input_wire not in edge:
                new_inputs.remove(input_wire)
    for output_wire in outputs:
        for edge in edges:
            if output_wire not in edge:
                new_outputs.remove(output_wire)
    return new_inputs, new_outputs


def fallback_node_to_matrix(node: Node, in_number: int, out_number: int) -> np.matrix:
    """fallback_node_to_matrix(node: Node, in_number: int, out_number: int) -> np.matrix

    This implementation of the algorithm can be used with matrix from numpy or matrix specified in :ref:`data`.
    Matrix implemented in **data** contain a *node_to_matrix* method but numpy matrix will use this fallback method.

    Args:
        node (Node): Node to be converted to matrix
        in_number (int): number of input to the node
        out_number (int): number of outputs to the node

    Returns:
        np.matrix: matrix representing the node given
    """
    if in_number < 0 or out_number < 0:
        raise ValueError('in_number and out_number must be positive integers')
    if node.node_type == 'hadamard':
        if in_number + out_number != 2:
            raise ValueError('Hadamard gate is only known for node with an arity of 2')
        result = np.ones((1 << out_number, 1 << in_number), dtype=complex)
        result[-1, -1] = -1
        result /= np.sqrt(2)
    elif node.node_type == 'Z':
        result = np.zeros((1 << out_number, 1 << in_number), dtype=complex)
        result[1, 1] = 1
        result[-1, -1] = np.exp(np.pi * node.angle * 1j)
    elif node.node_type == 'X':
        result = np.zeros((1 << out_number, 1 << in_number), dtype=complex)
        result[1, 1] = 1
        result[-1, -1] = np.exp(np.pi * node.angle * 1j)
        h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
        result = tensor_power(h, out_number).dot(result).dot(tensor_power(h, in_number))
    else:
        raise ValueError('Unknown node type: %s' % node.node_type)
    return result


def symmetric_difference(x: Iterable, y: Iterable) -> Iterable:
    """symmetric_difference(x: Iterable, y: Iterable) -> Iterable

    Symmetric difference between two iterables

    Args:
        x (Iterable): first iterable
        y (Iterable): second iterable

    Returns:
        Iterable: symmetric difference between *x* and *y*
    """
    return [i for i in x if i not in y] + [i for i in y if i not in x]


def tensor_product(a: GenericMatrix, b: GenericMatrix) -> GenericMatrix:
    """tensor_product(a: GenericMatrix, b: GenericMatrix) -> GenericMatrix

    Computes the tensor product of matrix *a* and *b*
    
    Args:
        a (GenericMatrix): First argument, have to be a matrix (numpy matrix or 2-D array)
        b (GenericMatrix): Second argument, have to be a matrix (numpy matrix or 2-D array)
    Returns:
        GenericMatrix: Tensor product of a and b
    """
    ma, na = a.shape
    mb, nb = b.shape
    mr, nr = ma * mb, na * nb
    result = UsedFragment(np.zeros((mr, nr), dtype=complex))
    for i in np.arange(mr):
        for j in np.arange(nr):
            result[i, j] = a[i // mb, j // nb] * b[i % mb, j % nb]
    return result


def tensor_power(a: GenericMatrix, power: int) -> GenericMatrix:
    """tensor_power(a: GenericMatrix, power: int) -> GenericMatrix

    Computes the *a**power*, in the tensor sense

    Args:
        a (GenericMatrix): First argument, has to be a matrix (numpy matrix or 2-D array)
        power (int): The power to elevate a to, must be positive (or equal to 0)
    Returns:
        GenericMatrix: 'power'
    """
    if power < 0:
        raise ValueError('Tensor power not defined for a negative power')
    if power == 0:
        return UsedFragment(np.identity(1))
    else:
        try:
            return a.tensor_product(a.tensor_power(power - 1))
        except AttributeError:
            return tensor_product(a, tensor_power(a, power - 1))
