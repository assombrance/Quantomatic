import cmath
import math
import collections
import numpy as np


def connected_graph_matrix(start_nodes, end_nodes, inside_nodes, edges):
    """Takes as input a ZX-calculus connected diagram and returns the corresponding matrix.

    This is the naive algorithm version: it isn't optimised whatsoever, so expect terrible performances.
    The algorithm runs ass follow :
    It has a set of nodes strictly growing with the time, this set has a corresponding matrix.
    This set is initialised with the *start_nodes*, and as long as all the nodes aren't in the set of calculated nodes,
    it adds as many neighbours as possible to the set and multiply their
    matrices to the set's matrix.
    This way of going through the diagram requires the diagram to have a non empty set of *start_nodes*

    Args:
        start_nodes (list[node]): Inputs for this configuration of the diagram (non empty !)
        end_nodes (list[node]): Outputs for this configuration of the diagram
        inside_nodes (list[node]): Transformation in the diagram
        edges (list[[node_name1, node_name2]]): List of the edges between each nodes

    Returns:
        matrix[complex]: Matrix representing the diagram
    """
    matrix = None
    circuit = start_nodes
    circuit_names = []
    for node in start_nodes:
        for node_name in node:
            circuit_names.append(node_name)
    pre_permutation_nodes_order = sort_nodes(circuit)
    pre_permutation_edges_order = []
    for node in pre_permutation_nodes_order:
        for node_name in node:
            for edge in edges:
                for edge_name in edge:
                    if node_name in edge[edge_name]:
                        pre_permutation_edges_order.append(edge)
    end_nodes_names = []
    for node in end_nodes:
        for node_name in node:
            end_nodes_names.append(node_name)
    while not end_detection_main_algo(circuit_names, inside_nodes):
        next_nodes_to_be_added, next_nodes_to_be_added_names, _ = neighbours(circuit, inside_nodes + end_nodes, edges)
        next_nodes_to_be_added, next_nodes_to_be_added_names = remove_incompatible_nodes(next_nodes_to_be_added,
                                                                                         next_nodes_to_be_added_names,
                                                                                         edges)
        post_permutation_nodes_order = sort_nodes(next_nodes_to_be_added)
        post_permutation_edges_order, \
            post_permutation_nodes_order = post_permutation_edge_order_management(post_permutation_nodes_order,
                                                                                  circuit_names, edges,
                                                                                  next_nodes_to_be_added_names)
        m = permutation_matrix(pre_permutation_edges_order, post_permutation_edges_order)
        if matrix is None:
            matrix = np.matrix(m, dtype=complex)
        else:
            matrix = np.dot(m, matrix)
        m = np.full((1, 1), 1, dtype=complex)
        computational_nodes = list(post_permutation_nodes_order)
        to_remove = []
        for node in computational_nodes:
            for node_name in node:
                if node_name in end_nodes_names:
                    to_remove.append(node)
        for node in to_remove:
            computational_nodes.remove(node)
        nodes_matrices = nodes_matrix(computational_nodes)
        for node_matrix in nodes_matrices:
            m = tensor_product(m, node_matrix)
        pre_permutation_nodes_name = []
        for node in start_nodes:
            for node_name in node:
                pre_permutation_nodes_name.append(node_name)
        inside_neighbours = set(circuit_names + next_nodes_to_be_added_names + pre_permutation_nodes_name) - set(
            end_nodes_names)
        for edge in edges:
            for edge_name in edge:
                if ((edge[edge_name][0] in circuit_names and edge[edge_name][1] not in inside_neighbours)
                        or (edge[edge_name][1] in circuit_names and edge[edge_name][0] not in inside_neighbours)):
                    m = tensor_product(m, np.identity(2, dtype=complex))
        matrix = np.dot(m, matrix)
        pre_permutation_nodes_order = post_permutation_nodes_order
        pre_permutation_edges_order = pre_permutation_edge_order_management(pre_permutation_nodes_order,
                                                                            edges,
                                                                            circuit_names,
                                                                            next_nodes_to_be_added_names)
        next_nodes_to_be_added, next_nodes_to_be_added_names = remove_end_nodes_neighbours(next_nodes_to_be_added,
                                                                                           next_nodes_to_be_added_names,
                                                                                           end_nodes_names)
        circuit = circuit + next_nodes_to_be_added
        circuit_names = circuit_names + next_nodes_to_be_added_names
    return matrix


def main_algo(start_nodes, end_nodes, inside_nodes, edges):
    """Takes as input a ZX-calculus diagram and returns the corresponding matrix.

    To compute the matrix corresponding to a diagram, this algorithm splits it into connected graphs, and feeds
    them to the **connected_graph_matrix** function.
    If a graph doesn't have any input, the outputs are given as inputs and the resulting matrix is transposed.
    If a graph has neither no inputs and no outputs, it is given to **scalar_matrix** before
    **connected_graph_matrix** for an operation allowing it to work (described in **scalar_matrix**).
    Once done, all the result matrices are multiplied with the **tensor_product** function.

    Args:
        start_nodes (list[node]): Inputs for this configuration of the diagram
        end_nodes (list[node]): Outputs for this configuration of the diagram
        inside_nodes (list[node]): Transformation in the diagram
        edges (list[[node_name1, node_name2]]): List of the edges between each nodes

    Returns:
        matrix[complex]: Matrix representing the diagram
    """
    connected_graphs = split_in_connected_graphs(start_nodes, end_nodes, inside_nodes, edges)
    matrix_list = []
    for graph in connected_graphs:
        if graph['start_nodes']:
            matrix_list.append(connected_graph_matrix(graph['start_nodes'], graph['end_nodes'],
                                                      graph['inside_nodes'], graph['edges']))
        elif graph['end_nodes']:
            matrix_list.append(np.matrix(connected_graph_matrix(graph['end_nodes'], [],
                                                                graph['inside_nodes'], graph['edges'])).transpose())
        else:
            matrix_list.append(scalar_matrix(graph['inside_nodes'], graph['edges']))
    result = [[1]]
    for matrix in matrix_list:
        result = tensor_product(result, matrix)
    # I/O order management
    inputs_list_connected_graphs = []
    outputs_list_connected_graphs = []
    for graph in connected_graphs:
        for node in graph['start_nodes']:
            for node_name in node:
                inputs_list_connected_graphs.append(node_name)
        for node in graph['end_nodes']:
            for node_name in node:
                outputs_list_connected_graphs.append(node_name)
    inputs_list_original_order = []
    outputs_list_original_order = []
    for node in start_nodes:
        for node_name in node:
            inputs_list_original_order.append(node_name)
    for node in end_nodes:
        for node_name in node:
            outputs_list_original_order.append(node_name)
    inputs_permutation_matrix = permutation_matrix(inputs_list_original_order, inputs_list_connected_graphs)
    outputs_permutation_matrix = permutation_matrix(outputs_list_connected_graphs, outputs_list_original_order)
    result = np.dot(outputs_permutation_matrix, result)
    result = np.dot(result, inputs_permutation_matrix)
    return result


def manage_ins_outs(wire_vertices_dictionary, undir_edges_dictionary, inputs_order_list: list,
                    outputs_order_list: list):
    """Looks for unlisted inputs and outputs, in case some are found, if listed as input or output in the file, add them
    otherwise, raise an error.

    Args:
        wire_vertices_dictionary (dictionary): raw input/output dictionary
        undir_edges_dictionary (dictionary): raw edges dictionary
        inputs_order_list (list[string]): user input list
        outputs_order_list (list[string]): user output list

    Returns:
        bool, list, list
    """
    assumed_order = False
    for node_name in wire_vertices_dictionary:
        if node_name not in inputs_order_list + outputs_order_list:
            for edge_name in undir_edges_dictionary:
                if undir_edges_dictionary[edge_name]['tgt'] == node_name or \
                        undir_edges_dictionary[edge_name]['src'] == node_name:
                    if 'data' in undir_edges_dictionary[edge_name] and \
                            'label' in undir_edges_dictionary[edge_name]['data']:
                        if 'input' in undir_edges_dictionary[edge_name]['data']['label']:
                            inputs_order_list.append(node_name)
                            assumed_order = True
                        elif 'output' in undir_edges_dictionary[edge_name]['data']['label']:
                            outputs_order_list.append(node_name)
                            assumed_order = True
                        else:
                            raise NameError('Not all wire edges in Inputs + Outputs (missing \'' + node_name + '\')')
                    else:
                        raise NameError('Not all wire edges in Inputs + Outputs (missing \'' + node_name + '\')')
    return assumed_order, inputs_order_list, outputs_order_list


def scalar_matrix(inside_nodes, edges):
    """Computes the matrix corresponding to a 'scalar' diagram (a diagram without any inputs nor outputs)

    This function is necessary because the **connected_graph_matrix** function needs inputs to start, to circumvent this
    problem, it adds a fake input on a random node (using the spider rule).
    If the node is green, the result matrix will be multiplied by [[1] [1]], otherwise, it will be multiplied by
    [[sqrt(2)] [0]]
    (those matrices are the one corresponding respectively to a green node with one output or a red node
    with one output).

    Args:
        inside_nodes: (list[node]): Transformation in the diagram
        edges: (list[[node_name1, node_name2]]): List of the edges between each nodes
    Returns:
         matrix:
    """
    root = inside_nodes[0]
    for root_name in root:
        if 'data' not in root[root_name] or root[root_name]['data']['type'] == 'Z':
            fake_entry = [[1], [1]]
        else:
            fake_entry = np.matrix([[1], [0]]) * math.sqrt(2)
        return np.dot(connected_graph_matrix([{'co': {}}], [], inside_nodes, edges + [{'edge_co': ['co', root_name]}]),
                      fake_entry)


def split_in_connected_graphs(start_nodes, end_nodes, inside_nodes, edges):
    """Takes as input a ZX-calculus diagram and returns a list of connected graphs.

    To achieve this, a random node not yet in a constructed connected graph is chosen as a root. From this point,
    **build_connected_graph** gathers all the nodes and edges connected to this root.
    The type graph here is a dictionary with the *start_nodes*, *end_nodes*, *inside_nodes*, *edges* gathered in it

    Args:
        start_nodes (list[node]): Inputs for this configuration of the diagram
        end_nodes (list[node]): Outputs for this configuration of the diagram
        inside_nodes (list[node]): Transformation in the diagram
        edges (list[[node_name1, node_name2]]): List of the edges between each nodes

    Returns:
        list[graph]: List of connected graphs
    """
    connected_graphs = []
    while not end_detection_connected_graphs(connected_graphs, start_nodes, end_nodes, inside_nodes):
        root = choose_root(inside_nodes, connected_graphs)
        graph = build_connected_graph(root, start_nodes, end_nodes, inside_nodes, edges)
        connected_graphs.append(graph)
    return connected_graphs


def build_connected_graph(root, start_nodes, end_nodes, inside_nodes, edges):
    """Builds a connected graph from root.

    Starts from a graph with just one *inside_node* and progressively grow up to the point it doesn't change
    anymore.
    The growth is managed by **augment_graph** as well as the change detection.

    Args:
        root (node): A node from the inside nodes list (not yet in a connected graph when called in the
            **split_in_connected_graphs** function
        start_nodes (list[node]): Inputs for this configuration of the diagram
        end_nodes (list[node]): Outputs for this configuration of the diagram
        inside_nodes (list[node]): Transformation in the diagram
        edges (list[[node_name1, node_name2]]): List of the edges between each nodes

    Returns:
        list[graph]: List of connected graphs
    """
    changed = True
    connected_graph = {'start_nodes': [], 'end_nodes': [], 'inside_nodes': [root], 'edges': []}
    while changed:
        connected_graph, changed = augment_graph(connected_graph, start_nodes, end_nodes, inside_nodes, edges)
    return connected_graph


def augment_graph(connected_graph, start_nodes, end_nodes, inside_nodes, edges):
    """Augments a graph with all its neighbours and edges connecting these neighbours to the graph and between them.
    Also returns if any modification has been made.

    Args:
        connected_graph (graph): To be exact, this graph doesn't have to be connected, but it is the use case we are
            exploiting so we can expect that.
        start_nodes (list[node]): Inputs for this configuration of the diagram
        end_nodes (list[node]): Outputs for this configuration of the diagram
        inside_nodes (list[node]): Transformation in the diagram
        edges (list[[node_name1, node_name2]]): List of the edges between each nodes

    Returns:
        graph, bool: *connected_graph* augmented with it's neighbours and the correct edges ass well as a boolean
            informing if a change as been applied to *connected_graph* or not.
    """
    node_set = start_nodes + end_nodes + inside_nodes
    initial_node_set = connected_graph['start_nodes'] + connected_graph['end_nodes'] + connected_graph['inside_nodes']
    nodes_neighbours, _, connection_edges = neighbours(initial_node_set, node_set, edges)
    connected_graph['edges'] = connected_graph['edges'] + connection_edges
    change_done = False
    for node in nodes_neighbours:
        for node_name in node:
            for start_node in start_nodes:
                for start_node_name in start_node:
                    if node_name == start_node_name:
                        connected_graph['start_nodes'].append(node)
                        change_done = True
            for end_node in end_nodes:
                for end_node_name in end_node:
                    if node_name == end_node_name:
                        connected_graph['end_nodes'].append(node)
                        change_done = True
            for inside_node in inside_nodes:
                for inside_node_name in inside_node:
                    if node_name == inside_node_name:
                        connected_graph['inside_nodes'].append(node)
                        change_done = True
    return connected_graph, change_done


def choose_root(inside_nodes, connected_graphs):
    """Returns a node from *inside_nodes* which is not in any graph from *connected_graphs*. Returns *None* if this set
    is empty.

    Args:
        inside_nodes (list[node]):
        connected_graphs (list[graph]):
    Returns:
        node:
    """
    remaining_nodes = list(inside_nodes)
    for connected_graph in connected_graphs:
        for node in connected_graph['inside_nodes']:
            if node in remaining_nodes:
                del remaining_nodes[remaining_nodes.index(node)]
    return remaining_nodes[0] if len(remaining_nodes) else None


def symmetric_difference(x: collections.Iterable, y: collections.Iterable):
    """Symmetric difference between two iterables

    Args:
        x (iterable): first iterable
        y (iterable): second iterable

    Returns:
        iterable: symmetric difference between *x* and *y*
    """
    return [i for i in x if i not in y] + [i for i in y if i not in x]


def end_detection_connected_graphs(connected_graphs, start_nodes, end_nodes, inside_nodes):
    """Detects if all the nodes are in the built *connected_graphs*.

    Needed for the *connected_graphs* construction.

    Args:
        connected_graphs (list[graph]): List of connected graphs
        start_nodes (list[node]): Global list of start nodes
        end_nodes(list[node]): Global list of end nodes
        inside_nodes(list[node]): Global list of inside nodes

    Returns:
        bool: True if the end is reached, meaning if no node from the global lists are left from the *connected_graphs*.
    """
    current_start_nodes = []
    current_end_nodes = []
    current_inside_nodes = []
    for connected_graph in connected_graphs:
        current_start_nodes = current_start_nodes + connected_graph['start_nodes']
        current_end_nodes = current_end_nodes + connected_graph['end_nodes']
        current_inside_nodes = current_inside_nodes + connected_graph['inside_nodes']
    return (not symmetric_difference(current_start_nodes, start_nodes) and
            not symmetric_difference(current_end_nodes, end_nodes) and
            not symmetric_difference(current_inside_nodes, inside_nodes))


def end_detection_main_algo(circuit_names, inside_nodes):
    """Detects if all the end nodes are in the circuit. If so, end is reached, returns True. Else, returns false

    Args:
        circuit_names (list[string]): List of all nodes' name in the circuit
        inside_nodes (list[node]): List of the inside nodes in the graph
    Returns:
         bool: Main loop end reached
    """
    for node in inside_nodes:
        for node_name in node:
            if node_name not in circuit_names:
                return False
    return True


def sort_nodes(nodes_list):
    """Sorts a nodes list in alphanumeric order.

    The format for a node must be **{node_name: data}**

    Args:
        nodes_list (list[node]): A nodes list
    Returns:
        list[node]: The sorted nodes list
    """
    nodes_dictionary = nodes_list_to_nodes_dictionary(nodes_list)
    node_name_list_gate = []
    node_name_list_wire = []
    for node_name in nodes_dictionary:
        if 'b' not in node_name:
            node_name_list_gate.append(node_name)
        else:
            node_name_list_wire.append(node_name)
    node_name_list_gate.sort()
    node_name_list_wire.sort()
    node_name_list = node_name_list_gate + node_name_list_wire

    nodes_list_sorted = []
    for node_name_sorted in node_name_list:
        nodes_list_sorted.append({node_name_sorted: nodes_dictionary[node_name_sorted]})
    return nodes_list_sorted


def remove_end_nodes_neighbours(next_nodes_to_be_added, next_nodes_to_be_added_names, end_nodes_names):
    """Removes the end nodes from the *next_nodes_to_be_added*.

    This function is needed just before the node matrix calculations. Removing those nodes from the list is the
    equivalent of delaying them to the next layer. It is needed because the end nodes don't have a corresponding matrix.

    Args:
        next_nodes_to_be_added (list[node]): List of node next added to the circuit.
        next_nodes_to_be_added_names (list[string]): List of names corresponding to the above list.
        end_nodes_names (list[string]): List of nodes' name to remove from the above list.

    Returns:
        list[node], list[string]: Lists cleared from the end nodes.
    """
    for node in next_nodes_to_be_added:
        for node_name in node:
            if node_name in end_nodes_names:
                del next_nodes_to_be_added[next_nodes_to_be_added.index(node)]
                del next_nodes_to_be_added_names[next_nodes_to_be_added_names.index(node_name)]
    return next_nodes_to_be_added, next_nodes_to_be_added_names


def pre_permutation_edge_order_management(start_nodes_order, edges, circuit_names, next_nodes_to_be_added_names):
    """Returns the edges ordered by node (and more if some edges are not used)

    Args:
        start_nodes_order (list[node]): List of nodes before the permutation
        edges (list[edge]): List of edges in the graph
        circuit_names (list[string]): List on nodes' name in the current built circuit
        next_nodes_to_be_added_names (list[string]): List of nodes to be added next (needed to determine the edges
            coming out of the circuit but not used yet)

    Returns:

    """
    start_edges_order = []
    for node in start_nodes_order:
        for node_name in node:
            for edge in edges:
                for edge_name in edge:
                    if (node_name == edge[edge_name][0] and edge[edge_name][1] not in circuit_names) or \
                            (node_name == edge[edge_name][1] and edge[edge_name][0] not in circuit_names):
                        start_edges_order.append(edge)
    for edge in edges:
        for edge_name in edge:
            if ((edge[edge_name][0] in circuit_names and
                 edge[edge_name][1] not in circuit_names + next_nodes_to_be_added_names)
                    or (edge[edge_name][1] in circuit_names and
                        edge[edge_name][0] not in circuit_names + next_nodes_to_be_added_names)):
                start_edges_order.append(edge)
    return start_edges_order


def post_permutation_edge_order_management(end_nodes_order, circuit_names, edges, next_nodes_to_be_added_names):
    """Adds incoming and outgoing edged to each node.
    Creates also the *end_edges_order* with the edges reaching the neighbours nodes first, and then the edges reaching
    future nodes.

    Args:
        end_nodes_order (list[node]): List of nodes after the permutation, needed to order the edges properly. Moreover,
            nodes will be tagged with incoming and outgoing edges to compute their matrix later on.
        circuit_names (list[string]): List of nodes in the current built circuit, needed to know if an edge is incoming
            or outgoing.
        edges (list[edge]): List of edges in the graph
        next_nodes_to_be_added_names (list[node]): list of nodes' name to be added next, used to add the edges joining
            the previous layer and a layer coming later on.

    Returns:
        list[edge], list[node]: The list of edges and nodes with the appropriate values.
    """
    end_edges_order = []
    for node in end_nodes_order:
        for node_name in node:
            node[node_name]['edge_in'] = []
            node[node_name]['edge_out'] = []
            for edge in edges:
                if edge not in end_edges_order:
                    for edge_name in edge:
                        if (edge[edge_name][0] == node_name and edge[edge_name][1] in circuit_names) or \
                                (edge[edge_name][1] == node_name and edge[edge_name][0] in circuit_names):
                            end_edges_order.append(edge)
                            node[node_name]['edge_in'].append(edge)
                        if (edge[edge_name][0] == node_name and edge[edge_name][1] not in circuit_names) or \
                                (edge[edge_name][1] == node_name and edge[edge_name][0] not in circuit_names):
                            node[node_name]['edge_out'].append(edge)
    for edge in edges:
        if edge not in end_edges_order:
            for edge_name in edge:
                if ((edge[edge_name][0] in circuit_names and
                     edge[edge_name][1] not in circuit_names + next_nodes_to_be_added_names)
                        or (edge[edge_name][1] in circuit_names and
                            edge[edge_name][0] not in circuit_names + next_nodes_to_be_added_names)):
                    end_edges_order.append(edge)
    return end_edges_order, end_nodes_order


def remove_incompatible_nodes(nodes_list, nodes_list_names, edges):
    """In the **connected_graph_matrix** function, it is impossible to compute the matrices from two nodes if they are
    in the same layer. To avoid this problem, if there is an edge between two nodes of a layer, one of the two is
    removed via the current function.

    Args:
        nodes_list (list[node]): List of nodes in the next layer to be added.
        nodes_list_names (list[string]): List of the names corresponding to the abode nodes.
        edges (list[edge]): List of edges in the graph.
    Returns:
        (list[node], list[node_names]): List of nodes cleared from incompatibilities.
    """
    for edge in edges:
        for edge_name in edge:
            if edge[edge_name][0] in nodes_list_names and edge[edge_name][1] in nodes_list_names:
                nodes_list_names.remove(edge[edge_name][0])
                for node in nodes_list:
                    for node_name in node:
                        if node_name == edge[edge_name][0]:
                            nodes_list.remove(node)
    return nodes_list, nodes_list_names


def neighbours(subset, main_set, edges):
    """Returns the neighbours of *subset* in *set* as well as the edges joining this *subset* to the neighbours.

    The format for a node must be {node_name: data} and similarly, for an edge : {edge_name: [node1, node2]}

    Args:
        subset (list[node]): The subset of which we are looking for the neighbours
        main_set (list[node]): The global set containing the subset and possibly more
        edges (list[(node_name,node_name)]): The list of relations between elements of the set
    Returns:
        list[node], list[string], list[edge]: A list (possibly empty) of elements in *set*, these are all the
        *neighbours* of *subset*, a list of the name of each node of the previous list and a list of the edges joining
        the *subset* to the *neighbours* and the *neighbours* between each other.
    """
    subset_dictionary = nodes_list_to_nodes_dictionary(subset)
    main_set_dictionary = nodes_list_to_nodes_dictionary(main_set)
    neighbours_set = []
    neighbours_set_names = []
    joining_edges = []
    for edge in edges:
        for edge_name in edge:
            if edge[edge_name][0] in subset_dictionary and edge[edge_name][1] not in subset_dictionary:
                joining_edges.append(edge)
                if edge[edge_name][1] not in neighbours_set_names:
                    neighbours_set.append({edge[edge_name][1]: main_set_dictionary[edge[edge_name][1]]})
                    neighbours_set_names.append(edge[edge_name][1])
            elif edge[edge_name][1] in subset_dictionary and edge[edge_name][0] not in subset_dictionary:
                joining_edges.append(edge)
                if edge[edge_name][0] not in neighbours_set_names:
                    neighbours_set.append({edge[edge_name][0]: main_set_dictionary[edge[edge_name][0]]})
                    neighbours_set_names.append(edge[edge_name][0])
    for edge in edges:
        for edge_name in edge:
            if edge[edge_name][0] in neighbours_set_names and edge[edge_name][1] in neighbours_set_names:
                joining_edges.append(edge)
    return neighbours_set, neighbours_set_names, joining_edges


def permutation_matrix(pre_permutation_list, post_permutation_list):
    """Returns the square matrix corresponding to the permutations of the qbits given by te args

    Args:
        pre_permutation_list (list[edge]): List of edges before the permutation
        post_permutation_list (list[edge]): List of edges after the permutation
    Returns:
        matrix[complex]: qbits permutation matrix
    """
    length = len(pre_permutation_list)
    n = pow(2, length)
    matrix = np.zeros((n, n), dtype=complex)
    permutation_dic = build_permutation_dictionary(pre_permutation_list, post_permutation_list)
    for i in np.arange(n):
        j = image_by_permutation(permutation_dic, i)
        matrix[i][j] = 1
    return matrix


def build_permutation_dictionary(pre_permutation_list, post_permutation_list):
    """Builds a permutation dictionary (representing the permutation function) from two list given, the first one pre
    permutation and the second one post permutation.

    Args:
        pre_permutation_list (list): List before permutation.
        post_permutation_list (list): List after permutation.

    Returns:
        dictionary: Permutation function under a dictionary representation.
    """
    length = len(pre_permutation_list)
    permutation_dic = {}
    for i in np.arange(length):
        for j in np.arange(length):
            if pre_permutation_list[length - 1 - i] == post_permutation_list[length - 1 - j]:
                permutation_dic[i] = j
                break
    return permutation_dic


def image_by_permutation(permutation_dictionary, n):
    """Computes the image of *n* by the permutation described by *permutation_dictionary*.

    *permutation_dictionary* format : {Fiber (int): Image (int)}

    Args:
        permutation_dictionary (dictionary): Dictionary representing the permutation function
        n (int): Fiber to the image looked for.

    Returns:
        int: Image of *n* by the permutation.
    """
    length = len(permutation_dictionary)
    image = {}
    for i in np.arange(length):
        image[i] = digit(n, permutation_dictionary[i])
    return binary_dictionary_to_int(image)


def binary_dictionary_to_int(binary_dictionary):
    """Translate a binary number given under it's binary shape via the *binary_dictionary* to an int.

    *binary_dictionary* format : {digit_number (int): value (bool)}

    Args:
        binary_dictionary (dictionary): an int given under it's binary representation with a dictionary

    Returns:
        int: the int value of the number
    """
    length = len(binary_dictionary)
    result = 0
    for i in np.arange(length):
        result += pow(2, i) * binary_dictionary[i]
    return result


def tensor_product(a, b):
    """Computes the tensor product of matrix *a* and *b*

    *a* and *b* have to be matrices (numpy matrix or 2-D array)

    Args:
        a (matrix): First argument
        b (matrix): Second argument
    Returns:
        matrix: Tensor product of a and b
    """
    matrix_a = np.array(a, dtype=complex)
    matrix_b = np.array(b, dtype=complex)
    ma, na = matrix_a.shape
    mb, nb = matrix_b.shape
    mr, nr = ma * mb, na * nb
    result = np.zeros((mr, nr), dtype=complex)
    for i in np.arange(mr):
        for j in np.arange(nr):
            result[i][j] = matrix_a[int(i / mb)][int(j / nb)] * matrix_b[i % mb][j % nb]
    return result


def tensor_power(a, power):
    """Computes the *a**power*, in the tensor sense

    *a* has to be a matrix (numpy matrix or 2-D array)
    *power* must be positive (or equal to 0)

    Args:
        a (matrix): First argument
        power (int): The power to elevate a to
    Returns:
        matrix: 'power'
    """
    if power < 0:
        raise NameError('Tensor power not defined for a negative power')
    if power == 0:
        return np.identity(1)
    else:
        return tensor_product(a, tensor_power(a, power - 1))


def nodes_matrix(nodes):
    """Returns the list of matrices corresponding to each node.

    This function is the only one that uses the registered angles from the file. Extra caution has to be taken since
    the format can be :
        - a number or fraction, then Pi is implied
        - a number or fraction multiplied by Pi, then, Pi has to be parsed (not done in native python)
    Moreover, the number registration format may be rethink to : accelerate calculations, save memory, have more \
    precise calculations

    Args:
        nodes (list[node]): Node list which will give one matrix for each node
    Returns:
        list[matrix[complex]]: List of matrices corresponding to each node
    """
    matrix_list = []
    for node_func in nodes:
        for node_name_func in node_func:
            n = len(node_func[node_name_func]['edge_in'])
            m = len(node_func[node_name_func]['edge_out'])
            matrix_func = np.zeros((pow(2, m), pow(2, n)), dtype=complex)
            matrix_func[0][0] = 1
            if 'data' not in node_func[node_name_func]:
                # Z node, angle 0
                matrix_func[pow(2, m) - 1][pow(2, n) - 1] = 1
            else:
                if node_func[node_name_func]['data']['type'] == 'Z':
                    # Z node, angle node_func[node_name_func]['data']['value']
                    alpha = node_func[node_name_func]['data']['value']
                    alpha.replace('Pi', '1')
                    alpha = float(eval(alpha))
                    matrix_func[0][0] = 1
                    matrix_func[pow(2, m) - 1][pow(2, n) - 1] = cmath.exp(math.pi * alpha * 1j)
                elif node_func[node_name_func]['data']['type'] == 'X':
                    # X node, angle node_func[node_name_func]['data']['value']
                    if node_func[node_name_func]['data']['value'] == '':
                        alpha = 0
                    else:
                        alpha = node_func[node_name_func]['data']['value']
                        alpha.replace('Pi', '1')
                        alpha = float(eval(alpha))
                    matrix_func[0][0] = 1
                    matrix_func[pow(2, m) - 1][pow(2, n) - 1] = cmath.exp(math.pi * alpha * 1j)
                    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
                    a = tensor_power(h, m)
                    b = tensor_power(h, n)
                    matrix_func = np.dot(a, matrix_func)
                    matrix_func = np.dot(matrix_func, b)
                elif node_func[node_name_func]['data']['type'] == 'hadamard':
                    # Hadamard
                    if n == 1 and m == 1:
                        matrix_func = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
                    elif n == 2 and m == 0:
                        matrix_func = np.matrix([[1, 1, 1, -1]]) / np.sqrt(2)
                    else:
                        raise NameError('Unhandled Hadamard configuration : (' + str(n) + ',' + str(m) + ') instead of '
                                                                                                         '(1,1) or '
                                                                                                         '(2,0)')
                else:
                    raise NameError('Unknown node type : ' + node_func[node_name_func]['data']['type'])
            matrix_list.append(matrix_func)
    return matrix_list


def nodes_dictionary_to_nodes_list(nodes_dictionary):
    """Converts a dictionary of nodes into a list of nodes

    A node format is as follows : **{node_name: data}**
    Note that for the input type, nodes contains a **s**, it is important since one dictionary will contain many nodes \
    whereas the list will only contain one node êr entry (even though the node in under the form of a dictionary)

    Args:
        nodes_dictionary (dict{nodes}): The nodes classed in a dictionary linking their name and their data
    Returns:
        list[node]: The list on the above nodes (node format in function description)
    """
    nodes_list = []
    for node_dictionary in nodes_dictionary:
        nodes_list.append({node_dictionary: nodes_dictionary[node_dictionary]})
    return nodes_list


def nodes_list_to_nodes_dictionary(nodes_list):
    """Converts a list of nodes into a dictionary of nodes

    A node format is as follows : **{node_name: data}**
    Note that for the output type, nodes contains a **s**, it is important since one dictionary will contain many nodes\
    whereas the list will only contain one node êr entry (even though the node in under the form of a dictionary)

    Args:
        nodes_list list(node): The list on the above nodes (node format in function description)
    Returns:
        dict{nodes}: The nodes classed in a dictionary linking their name and their data
    """
    nodes_dictionary = {}
    for node_list_func in nodes_list:
        for node_name in node_list_func:
            nodes_dictionary[node_name] = node_list_func[node_name]
    return nodes_dictionary


def digit_exchange(k, i, j):
    """Exchanges the digit i and j in the binary representation of k

    Here, digit 0 is the one corresponding to 2^0, digit 1 to 2^1, and so on ...

    Args:
        k (int): The target integer
        i (int): The first digit index
        j (int): The second digit index
    Returns:
        int: The value of k once the digits are exchanged
    """
    return k - pow(2, i) * digit(k, i) + pow(2, i) * digit(k, j) - pow(2, j) * digit(k, j) + pow(2, j) * digit(k, i)


def digit(k, i):
    """Computes the digit i for the integer k in its binary representation

    Here, digit 0 is the one corresponding to 2^0, digit 1 to 2^1, and so on ...

    Args:
        k (int): The integer for which the digit is needed
        i (int): The number of the digit
    Returns:
        int: The digit i from the integer k (0 or 1)
    """
    return int(np.floor(k / pow(2, i)) - 2 * np.floor(k / pow(2, i + 1)))
