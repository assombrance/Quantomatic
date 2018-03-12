import cmath
import math
import numpy as np


def connected_graph_matrix(start_nodes, end_nodes, inside_nodes, edges):
    """Takes as input a ZX-calculus diagram and returns the corresponding matrix.

    This is the naive algorithm version: it isn't optimised whatsoever, so expect terrible performances.
    The algorithm runs ass follow : it's initialised with the starting nodes, and as long as all the end-nodes
    aren't in the set of calculated nodes, it adds as many neighbours as possible to the set and multiply their
    matrices to the set's matrix.

    Args:
        start_nodes (list[node]): Inputs for this configuration of the diagram
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
                if node_name in edge:
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
        post_permutation_edges_order, post_permutation_nodes_order = post_permutation_edge_order_management(
            post_permutation_nodes_order, circuit_names, edges, next_nodes_to_be_added_names)
        m = permutation_matrix(pre_permutation_edges_order, post_permutation_edges_order)
        if matrix is None:
            matrix = np.matrix(m, dtype=complex)
        else:
            matrix = np.dot(m, matrix)
        m = np.full((1, 1), 1, dtype=complex)
        computational_nodes = list(post_permutation_nodes_order)
        for node in computational_nodes:
            for node_name in node:
                if node_name in end_nodes_names:
                    del computational_nodes[computational_nodes.index(node)]
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
            if ((edge[0] in circuit_names and edge[1] not in inside_neighbours)
                    or (edge[1] in circuit_names and edge[0] not in inside_neighbours)):
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
            print('find something to do here')
    print(matrix_list)
    result = [[1]]
    for matrix in matrix_list:  # TODO attention à l'ordre des entrées et sorties ici !
        result = tensor_product(result, matrix)
    return result


def split_in_connected_graphs(start_nodes, end_nodes, inside_nodes, edges):
    connected_graphs = []
    while not end_detection_connected_graphs(connected_graphs, start_nodes, end_nodes, inside_nodes):
        root = choose_root(inside_nodes, connected_graphs)
        graph = build_connected_graph(root, start_nodes, end_nodes, inside_nodes, edges)
        connected_graphs.append(graph)
    return connected_graphs


def build_connected_graph(root, start_nodes, end_nodes, inside_nodes, edges):
    changed = True
    connected_graph = {'start_nodes': [], 'end_nodes': [], 'inside_nodes': [root], 'edges': []}
    while changed:
        connected_graph, changed = augment_graph(connected_graph, start_nodes, end_nodes, inside_nodes, edges)
    return connected_graph


def augment_graph(connected_graph, start_nodes, end_nodes, inside_nodes, edges):
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
    remaining_nodes = list(inside_nodes)
    for connected_graph in connected_graphs:
        for node in connected_graph['inside_nodes']:
            if node in remaining_nodes:
                del remaining_nodes[remaining_nodes.index(node)]
    return remaining_nodes[0]


def symmetric_difference(x, y):
    return [i for i in x if i not in y] + [i for i in y if i not in x]


def end_detection_connected_graphs(connected_graphs, start_nodes, end_nodes, inside_nodes):
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
        circuit_names:
        inside_nodes:
    Returns:
         bool: End reached
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
    for node in next_nodes_to_be_added:
        for node_name in node:
            if node_name in end_nodes_names:
                del next_nodes_to_be_added[next_nodes_to_be_added.index(node)]
                del next_nodes_to_be_added_names[next_nodes_to_be_added_names.index(node_name)]
    return next_nodes_to_be_added, next_nodes_to_be_added_names


def pre_permutation_edge_order_management(start_nodes_order, edges, circuit_names, next_nodes_to_be_added_names):
    start_edges_order = []
    for node in start_nodes_order:
        for node_name in node:
            for edge in edges:
                if (node_name == edge[0] and edge[1] not in circuit_names) or \
                        (node_name == edge[1] and edge[0] not in circuit_names):
                    start_edges_order.append(edge)
    for edge in edges:
        if ((edge[0] in circuit_names and edge[1] not in circuit_names + next_nodes_to_be_added_names)
                or (edge[1] in circuit_names and edge[0] not in circuit_names + next_nodes_to_be_added_names)):
            start_edges_order.append(edge)
    return start_edges_order


def post_permutation_edge_order_management(end_nodes_order, circuit_names, edges, next_nodes_to_be_added_names):
    """Adds the data for each node of the ins and outs for each one.
    Creates also the *end_edges_order* with the edges reaching the neighbours nodes first, and then the edges reaching
    future nodes.

    :param end_nodes_order:
    :param circuit_names:
    :param edges:
    :param next_nodes_to_be_added_names:
    :return:
    """
    end_edges_order = []
    for node in end_nodes_order:
        for node_name in node:
            node[node_name]['edge_in'] = []
            node[node_name]['edge_out'] = []
            for edge in edges:
                if edge not in end_edges_order:
                    if (edge[0] == node_name and edge[1] in circuit_names) or \
                            (edge[1] == node_name and edge[0] in circuit_names):
                        end_edges_order.append(edge)
                        node[node_name]['edge_in'].append(edge)
                    if (edge[0] == node_name and edge[1] not in circuit_names) or \
                            (edge[1] == node_name and edge[0] not in circuit_names):
                        node[node_name]['edge_out'].append(edge)
    for edge in edges:
        if edge not in end_edges_order:
            if ((edge[0] in circuit_names and edge[1] not in circuit_names + next_nodes_to_be_added_names)
                    or (edge[1] in circuit_names and edge[0] not in circuit_names + next_nodes_to_be_added_names)):
                end_edges_order.append(edge)
    return end_edges_order, end_nodes_order


def remove_incompatible_nodes(nodes_list, nodes_list_names, edges):
    """

    Args:
        nodes_list:
        nodes_list_names:
        edges:
    Returns:
        (list[node], list[node_names]): tmp
    """
    for edge in edges:
        if edge[0] in nodes_list_names and edge[1] in nodes_list_names:
            nodes_list_names.remove(edge[0])
            for node in nodes_list:
                for node_name in node:
                    if node_name == edge[0]:
                        nodes_list.remove(node)
    return nodes_list, nodes_list_names


def neighbours(subset, main_set, edges):
    """Returns the neighbours of *subset* in *set*.

    The format for a node must be {node_name: data}

    Args:
        subset (list[node]): The subset of which we are looking for the neighbours
        main_set (list[node]): The global set containing the subset and possibly more
        edges (list[(node_name,node_name)]): The list of relations between elements of the set
    Returns:
        (list[node], list[node_names]): A list (possibly empty) of elements in *set*, these are all the neighbours of *subset*
    """
    subset_dictionary = nodes_list_to_nodes_dictionary(subset)
    main_set_dictionary = nodes_list_to_nodes_dictionary(main_set)
    neighbours_set = []
    neighbours_set_names = []
    joining_edges = []
    for edge in edges:
        if edge[0] in subset_dictionary and edge[1] not in subset_dictionary and edge[1] not in neighbours_set_names:
            neighbours_set.append({edge[1]: main_set_dictionary[edge[1]]})
            neighbours_set_names.append(edge[1])
            joining_edges.append(edge)
        elif edge[1] in subset_dictionary and edge[0] not in subset_dictionary and edge[0] not in neighbours_set_names:
            neighbours_set.append({edge[0]: main_set_dictionary[edge[0]]})
            neighbours_set_names.append(edge[0])
            joining_edges.append(edge)
    for edge in edges:
        if edge[0] in neighbours_set_names and edge[1] in neighbours_set_names:
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
    length = len(pre_permutation_list)
    permutation_dic = {}
    for i in np.arange(length):
        for j in np.arange(length):
            if pre_permutation_list[length - 1 - i] == post_permutation_list[length - 1 - j]:
                permutation_dic[i] = j
                break
    return permutation_dic


def image_by_permutation(permutation_dictionary, n):
    length = len(permutation_dictionary)
    image = {}
    for i in np.arange(length):
        image[i] = digit(n, permutation_dictionary[i])
    return binary_dictionary_to_int(image)


def binary_dictionary_to_int(binary_dictionary):
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
                if node_func[node_name_func]['data']['value'] == '':
                    # X node, angle 0
                    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
                    matrix_func[pow(2, m) - 1][pow(2, n) - 1] = 1
                    a = tensor_power(h, m)
                    b = tensor_power(h, n)
                    matrix_func = np.dot(a, matrix_func)
                    matrix_func = np.dot(matrix_func, b)
                elif node_func[node_name_func]['data']['type'] == 'Z':
                    # Z node, angle node_func[node_name_func]['data']['value']
                    alpha = node_func[node_name_func]['data']['value']
                    alpha.replace('Pi', '1')
                    alpha = float(eval(alpha))  # TODO : do it better, possible code injection here
                    matrix_func[0][0] = 1
                    matrix_func[pow(2, m) - 1][pow(2, n) - 1] = cmath.exp(math.pi * alpha * 1j)
                else:
                    # X node, angle node_func[node_name_func]['data']['value']
                    alpha = node_func[node_name_func]['data']['value']
                    alpha.replace('Pi', '1')
                    alpha = float(eval(alpha))  # TODO : do it better, possible code injection here
                    matrix_func[0][0] = 1
                    matrix_func[pow(2, m) - 1][pow(2, n) - 1] = cmath.exp(math.pi * alpha * 1j)
                    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
                    a = tensor_power(h, m)
                    b = tensor_power(h, n)
                    matrix_func = np.dot(a, matrix_func)
                    matrix_func = np.dot(matrix_func, b)
            matrix_list.append(matrix_func)
    return matrix_list


def nodes_dictionary_to_nodes_list(nodes_dictionary):  # maybe sort it ?
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
        int: The value of k once the digits are axchanged
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


def two_wires_permutation_matrix(i, j, n):
    """Computes the permutation matrix between two wires

    The wires' indexes are counted from last to first

    Args:
        i (int): First wire
        j (int): Second wire
        n (int): Total number of qbits
    Return:
        matrix[int]: the permutation matrix resulting from this swap
    """
    matrix_func = np.identity(pow(2, n))
    for k in np.arange(pow(2, n) - 1):
        digit_i = digit(k, i)
        digit_j = digit(k, j)
        if digit_i < digit_j:
            k2 = digit_exchange(k, i, j)
            matrix_func[k][k] = 0.
            matrix_func[k2][k2] = 0.
            matrix_func[k][k2] = 1.
            matrix_func[k2][k] = 1.
    return matrix_func
