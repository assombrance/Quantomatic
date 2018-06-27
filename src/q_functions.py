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
    """Recursive function taking in a graph and returning the corresponding matrix.

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
            # we rewrite graph1 and graph2 so they contain two parts of the current graph1
            if no_node_edges_detection(graph.edges):
                # degenerate cases, when a graph contains only wires
                # in this case, graph1 will contain the I/O connected to another I/O and graph2 will contain the rest
                graph2 = Graph(nodes=graph.nodes)

                graph2 += filter_edges_inputs_outputs_by_nodes(graph2.nodes, graph)
                graph1 = graph - graph2
            else:
                if graph.inputs:
                    graph1 = Graph(inputs=[graph.inputs[0]])
                    graph1.augment(graph)
                elif graph.nodes:
                    graph1 = Graph(nodes=[graph.nodes[0]])
                else:
                    raise RuntimeError('A graph with no node shouldn\'t enter in this branch')

                graph1 += graph1.neighbouring_i_o(graph)
                graph2 = graph - graph1

                in_between_edges = between_graphs_edges(graph1, graph2, graph)

                graph1.edges += in_between_edges

                in_between_wires = []
                for edge in in_between_edges:
                    in_between_wires.append(Wire(edge.name))

                graph1.outputs += in_between_wires
                graph2.inputs += in_between_wires

            first_half_matrix = split_and_reunite(graph1)
            second_half_matrix = split_and_reunite(graph2)

            inter_matrix_link = matrix_linker(graph1, graph2)
        else:
            first_half_matrix = split_and_reunite(graph1)
            second_half_matrix = split_and_reunite(graph2)
            inter_matrix_link = []

        input_connections = wires_to_connection_point_node_sorted(graph.inputs, graph.edges,
                                                                  graph1.nodes, graph2.nodes, False)
        output_connections = wires_to_connection_point_node_sorted(graph.outputs, graph.edges,
                                                                   graph1.nodes, graph2.nodes, True)

        return divide_conquer.fusion_matrices(first_half_matrix, second_half_matrix, input_connections,
                                              output_connections, inter_matrix_link)


def between_graphs_edges(graph1: Graph, graph2: Graph, containing_graph: Graph) -> List[Edge]:
    """returns the edges linking two sub-graphs of *containing_graph*.

    Args:
        graph1 (Graph): first graph
        graph2 (Graph): second graph
        containing_graph (Graph): graph containing the two others

    Returns:
        List[Edge]: edges of *containing_graph* between *graph1* and *graph2*
    """
    linking_edges = []
    for edge in containing_graph.edges:
        if set(graph1.wires).intersection(edge) and set(graph2.wires).intersection(edge):
            linking_edges.append(edge)
    return linking_edges


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


def matrix_linker(graph1: Graph, graph2: Graph) -> List[InterMatrixLink]:
    """Creates a list of *InterMatrixLink* from the **m1** to **m2**.

    Args:
        graph1 (Graph): graph corresponding to **m1**
        graph2 (Graph): graph corresponding to **m2**

    Returns:
        List[InterMatrixLink]: links between **m1** and **m2** as a *InterMatrixLink* list
    """
    inter_matrix_link = []
    for edge in set(graph1.edges).intersection(graph2.edges):
        wire = Wire(edge.name)

        is_out = wire in graph1.outputs
        is_in = wire in graph1.inputs
        if is_out:
            index = graph1.outputs.index(wire)
        elif is_in:
            index = graph1.inputs.index(wire)
        else:
            continue
        connection_point1 = ConnectionPoint(is_matrix_2=False, is_out=is_out, index=index)

        is_out = wire in graph2.outputs
        is_in = wire in graph2.inputs
        if is_out:
            index = graph2.outputs.index(wire)
        elif is_in:
            index = graph2.inputs.index(wire)
        else:
            continue
        connection_point2 = ConnectionPoint(is_matrix_2=True, is_out=is_out, index=index)

        link = InterMatrixLink(connection_point1, connection_point2)
        inter_matrix_link.append(link)
    return inter_matrix_link


def filter_edges_inputs_outputs_by_nodes(nodes: List[Node], containing_graph: Graph) -> Graph:
    """

    Since the node list is split in two, many edges and wires don't need to be considered for the next iteration of
    each half. Thus, the edges, inputs and outputs are filtered so they are not passed to the next iteration if they are
    not in relation with the given nodes

    Args:
        nodes (List[Node]): nodes to apply the filter from
        containing_graph (Graph): graph to be filtered

    Returns:
        List[Edge], List[Wire], List[Wire]: edges, inputs and outputs without the members not in relation with the given
        nodes
    """
    new_edges = containing_graph.edges[:]
    new_inputs = containing_graph.inputs[:]
    new_outputs = containing_graph.outputs[:]
    for edge in containing_graph.edges:
        if not set(edge).intersection(nodes):  # edge doesn't contain any node from the list
            new_edges.remove(edge)
            if set(edge).intersection(containing_graph.inputs):
                new_inputs.remove(list(set(edge).intersection(containing_graph.inputs))[0])
            if Wire(edge.name) in containing_graph.inputs:
                new_inputs.remove(Wire(edge.name))
            if set(edge).intersection(containing_graph.outputs):
                new_outputs.remove(list(set(edge).intersection(containing_graph.outputs))[0])
            if Wire(edge.name) in containing_graph.outputs:
                new_outputs.remove(Wire(edge.name))
        elif len(set(edge).intersection(nodes)) == 1 and Wire(edge.name) not in new_outputs and \
                not set(edge).intersection(containing_graph.inputs) and \
                not set(edge).intersection(containing_graph.outputs):
            new_wire = Wire(edge.name)
            new_outputs.append(new_wire)
    return Graph(edges=new_edges, inputs=new_inputs, outputs=new_outputs)


def filter_edges_inputs_outputs_by_nodes_negative(nodes: List[Node], containing_graph: Graph) -> Graph:
    """

    Since the node list is split in two, many edges and wires don't need to be considered for the next iteration of
    each half. Thus, the edges, inputs and outputs are filtered so they are not passed to the next iteration if they are
    in relation with the given nodes from the other half

    Args:
        nodes (List[Node]): nodes to apply the filter from
        containing_graph (Graph): graph to be filtered

    Returns:
        Graph: graph containing the edges, inputs and outputs without the members in relation with the given nodes
    """
    new_edges = containing_graph.edges[:]
    new_inputs = containing_graph.inputs[:]
    new_outputs = containing_graph.outputs[:]
    for edge in containing_graph.edges:
        if set(edge).intersection(nodes):  # edge doesn't contain any node from the list
            new_edges.remove(edge)
            if set(edge).intersection(containing_graph.inputs):
                new_inputs.remove(list(set(edge).intersection(containing_graph.inputs))[0])
            if Wire(edge.name) in containing_graph.inputs:
                new_inputs.remove(Wire(edge.name))
            if set(edge).intersection(containing_graph.outputs):
                new_outputs.remove(list(set(edge).intersection(containing_graph.outputs))[0])
            if Wire(edge.name) in containing_graph.outputs:
                new_outputs.remove(Wire(edge.name))
    return Graph(edges=new_edges, inputs=new_inputs, outputs=new_outputs)


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
