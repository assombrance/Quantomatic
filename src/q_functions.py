from typing import List

import collections
import numpy as np

import divide_conquer

from data import UsedFragment, Node, Wire, Edge, GenericMatrix, InterMatrixLink, ConnectionPoint


def split_and_reunite(nodes: List[Node], edges: List[Edge], inputs: List[Wire], outputs: List[Wire]):
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

        return divide_conquer.fusion_matrices(first_half_matrix, second_half_matrix, input_connections,
                                              output_connections, inter_matrix_link)


def wires_to_connection_point(wires: List[Wire], edges: List[Edge], nodes_group_1: List[Node],
                              nodes_group_2: List[Node], is_output: bool) -> List[ConnectionPoint]:
    connection_points = []  # type: List[ConnectionPoint]
    connection_points_dict = {}  # type: dict
    index = 0
    for node in nodes_group_1:
        for edge in edges:
            if node in edge:
                if set(edge).intersection(wires):
                    wire_index = wires.index(list(set(edge).intersection(wires))[0])
                    point = ConnectionPoint(is_matrix_2=False, is_out=is_output, index=index)
                    connection_points_dict[wire_index] = point
                    index += 1
                if edge.name in [wire.name for wire in wires]:
                    wire_index = wires.index(Wire(edge.name))
                    point = ConnectionPoint(is_matrix_2=False, is_out=is_output, index=index)
                    connection_points_dict[wire_index] = point
                    index += 1
    index = 0
    for node in nodes_group_2:
        for edge in edges:
            if node in edge:
                if set(edge).intersection(wires):
                    wire_index = wires.index(list(set(edge).intersection(wires))[0])
                    point = ConnectionPoint(is_matrix_2=True, is_out=not is_output, index=index)
                    connection_points_dict[wire_index] = point
                    index += 1
                if edge.name in [wire.name for wire in wires]:
                    wire_index = wires.index(Wire(edge.name))
                    point = ConnectionPoint(is_matrix_2=True, is_out=not is_output, index=index)
                    connection_points_dict[wire_index] = point
                    index += 1
    for i in np.arange(len(wires)):
        connection_points.append(connection_points_dict[i])
    return connection_points


def matrix_linker(m1_outputs: List[Wire], m2_outputs: List[Wire]) -> List[InterMatrixLink]:
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
        # TODO we may have here the loop problem ... to consult later
    return new_edges, new_inputs, new_outputs


def fallback_node_to_matrix(node: Node, in_number: int, out_number: int):
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
    """Symmetric difference between two iterables

    Args:
        x (iterable): first iterable
        y (iterable): second iterable

    Returns:
        iterable: symmetric difference between *x* and *y*
    """
    return [i for i in x if i not in y] + [i for i in y if i not in x]


def tensor_product(a: GenericMatrix, b: GenericMatrix):
    """Computes the tensor product of matrix *a* and *b*
    *a* and *b* have to be matrices (numpy matrix or 2-D array)
    Args:
        a (GenericMatrix): First argument
        b (GenericMatrix): Second argument
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
    """Computes the *a**power*, in the tensor sense
    *a* has to be a matrix (numpy matrix or 2-D array)
    *power* must be positive (or equal to 0)
    Args:
        a (GenericMatrix): First argument
        power (int): The power to elevate a to
    Returns:
        GenericMatrix: 'power'
    """
    if power < 0:
        raise NameError('Tensor power not defined for a negative power')
    if power == 0:
        return np.identity(1)
    else:
        return tensor_product(a, tensor_power(a, power - 1))
