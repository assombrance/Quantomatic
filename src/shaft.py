import json

import cmath
import math
import numpy as np

nodes_dictionary = json.loads('''{
"v7":{"annotation":{"coord":[-1.0,3.0]}},
"v2":{"data":{"type":"X","value":"3"},"annotation":{"coord":[0.5,3.0]}},
"v6":{"annotation":{"coord":[2.0,3.0]}},
"v5":{"data":{"type":"Z","value":"1"},"annotation":{"coord":[-1.0,-1.25]}},
"v8":{"annotation":{"coord":[-1.75,0.5]}},
"v1":{"data":{"type":"X","value":""},"annotation":{"coord":[0.5,-1.25]}},
"v4":{"annotation":{"coord":[2.0,-1.25]}},
"v0":{"data":{"type":"X","value":""},"annotation":{"coord":[-1.0,1.25]}},
"v11":{"data":{"type":"X","value":""},"annotation":{"coord":[-1.75,1.5]}},
"v9":{"annotation":{"coord":[2.0,1.25]}},
"v3":{"data":{"type":"X","value":""},"annotation":{"coord":[2.0,-0.25]}},
"v10":{"annotation":{"coord":[0.5,1.25]}}
}''')

edges_dictionary = json.loads('''{
"e6":{"src":"v10","tgt":"v2"},
"e18":{"src":"v6","tgt":"b1"},
"e12":{"src":"v1","tgt":"v8"},
"e1":{"src":"v4","tgt":"v1"},
"e5":{"src":"v0","tgt":"v10"},
"e7":{"src":"v2","tgt":"v9"},
"e17":{"src":"v2","tgt":"v6"},
"e0":{"src":"b3","tgt":"v4"},
"e20":{"src":"v11","tgt":"v8"},
"e4":{"src":"v5","tgt":"v0"},
"e8":{"src":"v9","tgt":"v1"},
"e16":{"src":"v0","tgt":"v6"},
"e3":{"src":"v5","tgt":"v1"},
"e10":{"src":"v9","tgt":"v3"},
"e14":{"src":"v7","tgt":"v2"},
"e9":{"src":"v1","tgt":"v10"},
"e15":{"src":"v7","tgt":"b0"},
"e2":{"src":"b2","tgt":"v5"},
"e11":{"src":"v4","tgt":"v3"},
"e19":{"src":"v7","tgt":"v11"}
}''')

start_dictionary = json.loads('''{
"b1":{"annotation":{"boundary":true,"coord":[2.0,4.5]}},
"b0":{"annotation":{"boundary":true,"coord":[-1.0,4.5]}}
}''')

end_dictionary = json.loads('''{
"b2":{"annotation":{"boundary":true,"coord":[-1.0,-2.75]}},
"b3":{"annotation":{"boundary":true,"coord":[2.0,-2.75]}}
}''')

c = [
    {'v0': {
        'annotation': {'coord': [0.5, 1.5]},
        'edge_in': [['v0', 'v2']],
        'data': {'value': '3/4', 'type': 'X'},
        'edge_out': [['v3', 'v0'], ['v6', 'v0']]}},
    {'v6': {
        'annotation': {'coord': [2.5, 3.5]},
        'edge_in': [['b1', 'v6'], ['v6', 'v2']],
        'data': {'value': '1', 'type': 'Z'},
        'edge_out': [['v6', 'v0']]}},
    {'v9': {'annotation': {'coord': [1.5, 1.25]},
            'edge_in': [['v9', 'v2']],
            'data': {'value': '1/2', 'type': 'Z'},
            'edge_out': [['v1', 'v9']]}
     }]

m = [[0.70710678 + 0.j, 0. + 0.j],
     [0.70710678 + 0.j, 0. + 0.j],
     [0. + 0.j,         -0.5 + 0.5j],
     [0. + 0.j,         0.5 - 0.5j]]

# print(nodes_dictionary)
# print(edges_dictionary)

node_name_list = []
for node in nodes_dictionary:
    node_name_list.append(node)

node_name_list.sort(reverse=True)
# print(node_name_list)

node_list = []
for node_name in node_name_list:
    node_list.append({node_name: nodes_dictionary[node_name]})

# print(node_list)
edges = []
for edge_dictionary in edges_dictionary:
    edge = []
    for end in edges_dictionary[edge_dictionary]:
        edge.append(edges_dictionary[edge_dictionary][end])
    edges.append(edge)


# print(edges)


def sort_nodes(node_list_func):
    """Sort function

    :param node_list_func: input
    :return: node_list_sorted (output)
    """
    node_name_list_func = []
    for node_func in node_list_func:
        for node_name_func in node_func:
            node_name_list_func.append(node_name_func)
    node_name_list_func.sort()

    node_list_sorted = []
    for node_name_sorted in node_name_list_func:
        for node_func in node_list_func:
            for node_name_func in node_func:
                if node_name_func == node_name_sorted:
                    node_list_sorted.append(node_func)
    print(node_list_sorted)


# sort_nodes(node_list)


def nodes_dictionary_to_nodes_list(nodes_dictionary_func2):  # maybe sort it ?
    """

    :param nodes_dictionary_func2:
    :return:
    """
    nodes_list = []
    for node_dictionary in nodes_dictionary_func2:
        nodes_list.append({node_dictionary: nodes_dictionary_func2[node_dictionary]})
    return nodes_list


def nodes_list_to_nodes_dictionary(nodes_list_func):
    """

    :param nodes_list_func:
    :return:
    """
    nodes_dictionary_func2 = {}
    for node_list_func in nodes_list_func:
        for node_name_func in node_list_func:
            nodes_dictionary_func2[node_name_func] = node_list_func[node_name_func]
    return nodes_dictionary_func2


subset = node_list
main_set = nodes_dictionary_to_nodes_list(start_dictionary) + node_list + nodes_dictionary_to_nodes_list(end_dictionary)


def neighbours(subset_func, main_set_func, edges_func):
    subset_dictionary = nodes_list_to_nodes_dictionary(subset_func)
    print(subset_dictionary)
    main_set_dictionary = nodes_list_to_nodes_dictionary(main_set_func)
    print(main_set_dictionary)
    neighbours_set = []
    for edge_func in edges_func:
        if edge_func[0] in subset_dictionary and edge_func[1] not in subset_dictionary:
            neighbours_set.append({edge_func[1]: main_set_dictionary[edge_func[1]]})
        elif edge_func[1] in subset_dictionary and edge_func[0] not in subset_dictionary:
            neighbours_set.append({edge_func[0]: main_set_dictionary[edge_func[0]]})
    return neighbours_set


# print(neighbours(subset, main_set, edges))

start_edges_order = list(edges)
# print(start_edges_order)
del start_edges_order[4:]
end_edges_order = list(start_edges_order)
end_edges_order.reverse()
# print(start_edges_order)
# print(end_edges_order)

start_edges_order2 = list(start_edges_order)
del start_edges_order2[3:]
end_edges_order2 = list(start_edges_order2)
end_edges_order2[0], end_edges_order2[1], end_edges_order2[2] = end_edges_order2[2], end_edges_order2[0], \
                                                                end_edges_order2[1]


# print(start_edges_order2)
# print(end_edges_order2)


def digit_exchange(k, i_func, j):
    return k - pow(2, i_func) * digit(k, i_func) + pow(2, i_func) * digit(k, j) - pow(2, j) * digit(k, j) + \
           pow(2, j) * digit(k, i_func)


def digit(k, i_func):
    return int(np.floor(k / pow(2, i_func)) - 2 * np.floor(k / pow(2, i_func + 1)))
    # return int(bin(k)[-i_func]) : other possibility


# for i in np.arange(5):
#     print(i, " : ", digit(8, i))


def two_wire_permutation_matrix(i_func, j, n):
    matrix_func = np.identity(pow(2, n))
    for k in np.arange(pow(2, n) - 1):
        digit_i = digit(k, i_func)
        digit_j = digit(k, j)
        if digit_i < digit_j:
            k2 = digit_exchange(k, i_func, j)
            matrix_func[k][k] = 0.
            matrix_func[k2][k2] = 0.
            matrix_func[k][k2] = 1.
            matrix_func[k2][k] = 1.
    return matrix_func


# print(two_wire_permutation_matrix(0, 2, 3))


def list_permutation_matrix(pre_permutation_list, post_permutation_list):
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
                    permutation = two_wire_permutation_matrix(i, j, n)
                    matrix = np.dot(matrix, permutation)
                    post_permutation_list[i], post_permutation_list[j] = post_permutation_list[j], \
                                                                         post_permutation_list[i]
    return matrix


# print(np.linalg.matrix_power(list_permutation_matrix(start_edges_order2, end_edges_order2), 1))
def tensor_product(a, b):
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
    result = np.identity(a.shape[0])
    for _ in np.arange(power):
        result = tensor_product(result, a)
    return result


def nodes_matrix(nodes):
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


del node_list[:4]
del node_list[4:]
node_list[0]['v5']['edge_in'] = np.arange(3)
node_list[0]['v5']['edge_out'] = np.arange(2)
node_list[1]['v4']['edge_in'] = np.arange(2)
node_list[1]['v4']['edge_out'] = np.arange(4)
node_list[2]['v3']['edge_in'] = np.arange(3)
node_list[2]['v3']['edge_out'] = np.arange(2)
node_list[3]['v2']['edge_in'] = np.arange(2)
node_list[3]['v2']['edge_out'] = np.arange(4)

np.set_printoptions(linewidth=200)
for matrix in nodes_matrix(node_list):
    print(matrix)
    print('_________________________________________________________________________________________________'
          '_________________________________________________________________________________________________')
# node_list[1]['v8']['edge_in'] = []
# for edge in edges:
#     # print(edge)
#     if 'v8' in edge:
#         node_list[1]['v8']['edge_in'].append(edge)
# print(node_list[1]['v8']['edge_in'])
# print(len(node_list[1]['v8']['edge_in']))

test = [1, 2, 3, 4, 5]
print(test)
for number in test:
    if number == 3:
        del test[test.index(3)]
print(test)
