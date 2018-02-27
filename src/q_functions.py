import cmath
import numpy as np


def main_algo(start_nodes, end_nodes, inside_nodes, edges):
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
    start_nodes_order = sort_nodes(circuit)
    start_edges_order = []
    for node in start_nodes_order:
        for edge in edges:
            if node in edge:
                start_edges_order.append(edge)
    while end_nodes not in circuit:
        next_nodes_to_be_added = neighbours(circuit, inside_nodes + end_nodes, edges)
        for edge in edges:
            if edge[0] in next_nodes_to_be_added and edge[1] in next_nodes_to_be_added:
                next_nodes_to_be_added.remove(edge[0])
        end_nodes_order = sort_nodes(next_nodes_to_be_added)
        end_edges_order = []
        for node in end_nodes_order:
            for edge in edges:  # TODO attention, probablement des bugs de format ici
                if (edge[0] == node and edge[1] in circuit) or (edge[1] == node and edge[0] in circuit):
                    end_edges_order.append(edge)
                    node['edge_in'].append(edge)
                if (edge[0] == node and edge[1] not in circuit) or (edge[1] == node and edge[0] not in circuit):
                    node['edge_out'].append(edge)
            for edge in edges:
                if ((edge[0] in circuit and edge[1] not in circuit + next_nodes_to_be_added)
                        and (edge[1] in circuit and edge[0] not in circuit + next_nodes_to_be_added)):
                    end_edges_order.append(edge)
        m = permutation_matrix(start_edges_order, end_edges_order)
        if matrix is None:
            matrix = m
        else:
            matrix = np.dot(matrix, m)
        m = 1
        # TODO : penser à retirer les sorties
        nodes_matrices = nodes_matrix(next_nodes_to_be_added)
        for node_matrix in nodes_matrices:
            m = np.tensordot(m, node_matrix)
        # TODO : penser à ajouter les fils allant vers les sorties
        for edge in edges:
            if ((edge[0] in circuit and edge[1] not in circuit + next_nodes_to_be_added)
                    and (edge[1] in circuit and edge[0] not in circuit + next_nodes_to_be_added)):
                m = np.tensordot(m, np.identity(2))
        matrix = np.dot(matrix, m)
        start_nodes_order = end_nodes_order
        start_edges_order = []
        for node in start_nodes_order:
            for edge in edges:
                if node in edge:
                    start_edges_order.append(edge)
        for edge in edges:
            if ((edge[0] in circuit and edge[1] not in circuit + next_nodes_to_be_added)
                    and (edge[1] in circuit and edge[0] not in circuit + next_nodes_to_be_added)):
                start_edges_order.append(edge)
        circuit = circuit + next_nodes_to_be_added
    return matrix


def sort_nodes(nodes_list):
    """Sorts a nodes list in alphanumeric order.

    The format for a node must be **{node_name: data}**

    Args:
        nodes_list (list[node]): A nodes list
    Returns:
        list[node]: The sorted nodes list
    """
    nodes_dictionary = nodes_list_to_nodes_dictionary(nodes_list)
    node_name_list = []
    for node_name in nodes_dictionary:
        node_name_list.append(node_name)
    node_name_list.sort()

    nodes_list_sorted = []
    for node_name_sorted in node_name_list:
        nodes_list_sorted.append({node_name_sorted: nodes_dictionary[node_name_sorted]})
    return nodes_list_sorted


def neighbours(subset, main_set, edges):
    """Returns the neighbours of *subset* in *set*.

    The format for a node must be {node_name: data}

    Args:
        subset (list[node]): The subset of which we are looking for the neighbours
        main_set (list[node]): The global set containing the subset and possibly more
        edges (list[(node_name,node_name)]): The list of relations between elements of the set
    Returns:
        list[node]: A list (possibly empty) of elements in *set*, these are all the neighbours of *subset*
    """
    subset_dictionary = nodes_list_to_nodes_dictionary(subset)
    print(subset_dictionary)
    main_set_dictionary = nodes_list_to_nodes_dictionary(main_set)
    print(main_set_dictionary)
    neighbours_set = []
    for edge in edges:
        if edge[0] in subset_dictionary and edge[1] not in subset_dictionary:
            neighbours_set.append({edge[1]: main_set_dictionary[edge[1]]})
        elif edge[1] in subset_dictionary and edge[0] not in subset_dictionary:
            neighbours_set.append({edge[0]: main_set_dictionary[edge[0]]})
    return neighbours_set


def permutation_matrix(pre_permutation_list, post_permutation_list):
    """Returns the square matrix corresponding to the permutations of the qbits given by te args

    Args:
        pre_permutation_list (list[edge]): List of edges before the permutation
        post_permutation_list (list[edge]): List of edges after the permutation
    Returns:
        matrix[complex]: qbits permutation matrix
    """
    n = len(pre_permutation_list)
    matrix = np.identity(pow(2, n))
    for i in np.arange(n):
        if pre_permutation_list[i] == post_permutation_list[i]:
            continue
        else:
            for j in np.arange(i + 1, n):
                if pre_permutation_list[i] != post_permutation_list[j]:
                    continue
                else:
                    permutation = two_wires_permutation_matrix(i, j, n)
                    matrix = np.dot(matrix, permutation)
                    post_permutation_list[i], post_permutation_list[j] = post_permutation_list[j], \
                        post_permutation_list[i]
    return matrix


def tensor_product(a, b):
    """Computes the tensor product of matrix *a* and *b*

    *a* and *b* have to be matrices (numpy matrix or 2-D array)

    Args:
        a (matrix): First argument
        b (matrix): Second argument
    Returns:
        matrix: Tensor product of a and b
    """
    matrix_a = np.array(a)
    matrix_b = np.array(b)
    ma, na = matrix_a.shape
    mb, nb = matrix_b.shape
    mr, nr = ma * mb, na * nb
    result = np.zeros((mr, nr))
    for i in np.arange(mr):
        for j in np.arange(nr):
            result[i][j] = matrix_a[int(i / mb)][int(j / nb)] * matrix_b[i % mb][j % nb]
    return result


def tensor_power(a, power):
    """Computes the *a**power*, in the tensor sense

    *a* has to be a matrix (numpy matrix or 2-D array)

    Args:
        a (matrix): First argument
        power (int): The power to elevate a to
    Returns:
        matrix: 'power'
    """
    result = np.identity(a.shape[0])
    for _ in np.arange(power):
        result = tensor_product(result, a)
    return result


def nodes_matrix(nodes):  # TODO check if the calculus are correct
    """Returns the list of matrices corresponding to each node

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
                    matrix_func = np.dot(tensor_power(h, m - 1), np.dot(matrix_func, tensor_power(h, n - 1)))
                elif node_func[node_name_func]['data']['type'] == 'Z':
                    # Z node, angle node_func[node_name_func]['data']['value']
                    alpha = float(node_func[node_name_func]['data']['value'])
                    matrix_func[0][0] = 1
                    matrix_func[pow(2, m) - 1][pow(2, n) - 1] = cmath.exp(alpha * 1j)
                else:
                    # X node, angle node_func[node_name_func]['data']['value']
                    alpha = float(node_func[node_name_func]['data']['value'])
                    matrix_func[0][0], matrix_func[pow(2, m) - 1][pow(2, n) - 1] = 1, cmath.exp(alpha * 1j)
                    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
                    matrix_func = np.dot(tensor_power(h, m - 1), np.dot(matrix_func, tensor_power(h, n - 1)))
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
