# coding=UTF-8
"""
Module focused on going through the graph.
"""
from typing import List

import collections
import numpy as np

import divide_conquer

from data import UsedFragment, Node, Wire, Edge, GenericMatrix, InterMatrixLink, ConnectionPoint


def split_and_reunite(nodes: List[Node], edges: List[Edge], inputs: List[Wire], outputs: List[Wire]) -> np.matrix:
    """
    Recursive function taking in a graph and returning the corresponding matrix.

    To do so, split the graph in two, passes the two halves to it's next iteration and reunite the two matrix obtained
    using the :ref:`fusion_matrices <fusion_matrices>` method from :ref:`divide_conquer`.

    The main part of this function is converting the graph format to the matrix format. tmp

    Args:
        nodes:
        edges:
        inputs:
        outputs:

    Returns:

    """
    # TODO problem if edges not connected to a node (input cup, output cap or input-output edge)
    # for now, let's assume this problem won't happen and we'll figure it out later
    if len(nodes) == 1:
        try:
            return UsedFragment.node_to_matrix(nodes[0], len(inputs), len(outputs))
        except AttributeError:
            return fallback_node_to_matrix(nodes[0], len(inputs), len(outputs))
    else:
        half = len(nodes) // 2
        first_half_nodes = nodes[:half]
        second_half_nodes = nodes[half:]

        first_half_edges, first_half_inputs, first_half_outputs = filter_edges_inputs_outputs(first_half_nodes,
                                                                                              edges, inputs, outputs)
        second_half_edges, second_half_inputs, second_half_outputs = filter_edges_inputs_outputs(second_half_nodes,
                                                                                                 edges, inputs, outputs)

        first_half_matrix = split_and_reunite(first_half_nodes, first_half_edges,
                                              first_half_inputs, first_half_outputs)
        second_half_matrix = split_and_reunite(second_half_nodes, second_half_edges,
                                               second_half_inputs, second_half_outputs)

        inter_matrix_link = matrix_linker(first_half_outputs, second_half_outputs)
        input_connections = wires_to_connection_point(inputs, edges, first_half_nodes, second_half_nodes, False)
        output_connections = wires_to_connection_point(outputs, edges, first_half_nodes, second_half_nodes, True)
        # TODO I done shit ... output connexion should be calculated from first_half_outputs and second_half_outputs,
        # TODO here, I'm trying to reinvent the wheel and i'm risking de-synchronisation between the outputs
        # (same for inputs)

        return divide_conquer.fusion_matrices(first_half_matrix, second_half_matrix, input_connections,
                                              output_connections, inter_matrix_link)


def wires_to_connection_point(wires: List[Wire], edges: List[Edge], nodes_group_1: List[Node],
                              nodes_group_2: List[Node], is_output: bool) -> List[ConnectionPoint]:
    """
    Part of the conversion system from the data under the graph form (nodes, wires and edges) to the matrix form
    (links, connection points). In particular, Does the *ConnectionPoint* part.

    Args:
        wires (List[Wire]): list of the considered wires in the diagram
        edges (List[Edge]: list of all the edges in the diagram
        nodes_group_1 (List[Node]): list of the first half of the nodes in the diagram
        nodes_group_2 (List[Node]): list of the first half of the nodes in the diagram
        is_output (bool): are the wires given outputs or inputs

    Returns:

    """
    connection_points = []  # type: List[ConnectionPoint]
    connection_points_dict = {}  # type: dict

    def _wires_to_connection_point(_wires, _edges, _nodes_group, _is_output, _is_matrix_2, _cp_dict):
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

    connection_points_dict = _wires_to_connection_point(wires, edges, nodes_group_1, is_output,
                                                        False, connection_points_dict)
    connection_points_dict = _wires_to_connection_point(wires, edges, nodes_group_2, is_output,
                                                        True, connection_points_dict)
    for i in np.arange(len(wires)):
        connection_points.append(connection_points_dict[i])
    return connection_points


def matrix_linker(m1_outputs: List[Wire], m2_outputs: List[Wire]) -> List[InterMatrixLink]:
    """
    Creates a list of *InterMatrixLink* from the common outputs of **m1** and **m2**.

    Links between the two matrices are forced to be between their outputs, that's why you don't need the inputs.

    Args:
        m1_outputs (List[Wire]): outputs from **m1** as a *Wire* list
        m2_outputs (List[Wire]): outputs from **m2** as a *Wire* list

    Returns:
        List[InterMatrixLink]: links between **m1** and **m2** as a *InterMatrixLink* list
    """
    inter_matrix_link = []  # type: List[InterMatrixLink]
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


def filter_edges_inputs_outputs(nodes: List[Node], edges: List[Edge],
                                inputs: List[Wire], outputs: List[Wire]) -> (List[Edge], List[Wire], List[Wire]):
    """
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


def fallback_node_to_matrix(node: Node, in_number: int, out_number: int) -> np.matrix:
    """
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


def symmetric_difference(x: collections.Iterable, y: collections.Iterable):
    """
    Symmetric difference between two iterables

    Args:
        x (iterable): first iterable
        y (iterable): second iterable

    Returns:
        iterable: symmetric difference between *x* and *y*
    """
    return [i for i in x if i not in y] + [i for i in y if i not in x]


def tensor_product(a: GenericMatrix, b: GenericMatrix):
    """
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


def tensor_power(a: GenericMatrix, power: int):
    """
    Computes the *a**power*, in the tensor sense

    Args:
        a (GenericMatrix): First argument, has to be a matrix (numpy matrix or 2-D array)
        power (int): The power to elevate a to, must be positive (or equal to 0)
    Returns:
        GenericMatrix: 'power'
    """
    if power < 0:
        raise NameError('Tensor power not defined for a negative power')
    if power == 0:
        return np.identity(1)
    else:
        return tensor_product(a, tensor_power(a, power - 1))
