import json
import numpy as np

import q_functions as qf


def test_digit_2nd_digit_of_11():
    """
        11 = 1 0 1 1
    indexes: 3,2,1,0
    """
    assert qf.digit(11, 1) == 1


def test_digit_exchange_5_exchange_0_and_1():
    """
         5 = 1 0  1
    indexes: 2,1,0
    """
    assert qf.digit_exchange(5, 0, 1) == 6


def test_sort_nodes_enabled():
    """
    we check that the list coming from sort_nodes id sorted
    """
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
    node_name_list = []
    for node in nodes_dictionary:
        node_name_list.append(node)

    node_name_list.sort(reverse=True)
    # print(node_name_list)

    node_list = []
    for node_name in node_name_list:
        node_list.append({node_name: nodes_dictionary[node_name]})

    sorted_node_list = qf.sort_nodes(node_list)
    # sorted_node_list = node_list

    for i in np.arange(len(sorted_node_list)-1):
        for node_name1 in sorted_node_list[i]:
            for node_name2 in sorted_node_list[i+1]:
                assert node_name1 <= node_name2


def test_sort_nodes_disabled():
    """
    For the same list, we check that it is not sorted to begin with
    """
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
    node_name_list = []
    for node in nodes_dictionary:
        node_name_list.append(node)

    node_name_list.sort(reverse=True)
    # print(node_name_list)

    node_list = []
    for node_name in node_name_list:
        node_list.append({node_name: nodes_dictionary[node_name]})

    # sorted_node_list = qf.sort_nodes(node_list)
    sorted_node_list = node_list

    sorted_bool = True
    for i in np.arange(len(sorted_node_list)-1):
        for node_name1 in sorted_node_list[i]:
            for node_name2 in sorted_node_list[i+1]:
                if node_name1 > node_name2:
                    sorted_bool = False
    assert not sorted_bool


def test_remove_incompatible_nodes():
    next_nodes_to_be_added = [{'v0': {'data': {'type': 'X', 'value': '3*Pi/4'}, 'annotation': {'coord': [0.5, 1.25]}}},
                              {'v2': {'data': {'type': 'X', 'value': ''}, 'annotation': {'coord': [0.5, 2.5]}}}]
    next_nodes_to_be_added_names = ['v0', 'v2']
    edges = [['v0', 'v2'], ['v6', 'v2'], ['b1', 'v6'], ['v5', 'b2'], ['v3', 'v0'], ['v1', 'v9'], ['v1', 'b3'],
             ['v9', 'v2'], ['v1', 'v5'], ['v3', 'v5'], ['v6', 'v0'], ['v2', 'b0']]
    result = [['v0'], ['v2']]
    _, next_nodes_to_be_added_names = qf.remove_incompatible_nodes(next_nodes_to_be_added,
                                                                   next_nodes_to_be_added_names, edges)
    assert next_nodes_to_be_added_names in result


def test_neighbours():
    circuit = [{'b0': {'annotation': {'coord': [0.5, 4.5], 'boundary': True}}},
               {'b1': {'annotation': {'coord': [2.5, 4.5], 'boundary': True}}},
               {'v6': {
                   'edge_out': [['v2', 'v6'], ['v0', 'v6']],
                   'edge_in': [['v6', 'b1']], 'data': {'type': 'Z', 'value': '1'},
                   'annotation': {'coord': [2.5, 3.5]}}}]
    inside_nodes = [{'v5': {'annotation': {'coord': [0.5, -1.25]}}},
                    {'v3': {'annotation': {'coord': [0.5, 0.0]}, 'data': {'type': 'hadamard', 'value': ''}}},
                    {'v2': {
                        'edge_in': [['v2', 'b0']],
                        'edge_out': [['v0', 'v2'], ['v6', 'v2'], ['v9', 'v2']],
                        'annotation': {'coord': [0.5, 2.5]},
                        'data': {'type': 'X', 'value': ''}}},
                    {'v9': {'annotation': {'coord': [2.0, 1.25]}, 'data': {'type': 'Z', 'value': '1/2'}}},
                    {'v6': {'annotation': {'coord': [2.0, 3.5]}, 'data': {'type': 'Z', 'value': '1'}}},
                    {'v1': {'annotation': {'coord': [2.0, 0.0]}, 'data': {'type': 'X', 'value': ''}}},
                    {'v0': {'annotation': {'coord': [0.5, 1.25]}, 'data': {'type': 'X', 'value': '3/4'}}}]
    end_nodes = [{'b3': {'annotation': {'boundary': True, 'coord': [2.0, -2.75]}}},
                 {'b2': {'annotation': {'boundary': True, 'coord': [0.5, -2.75]}}}]
    edges = [['v0', 'v2'], ['v6', 'v2'], ['b1', 'v6'], ['v5', 'b2'], ['v3', 'v0'], ['v1', 'v9'], ['v1', 'b3'],
             ['v9', 'v2'], ['v1', 'v5'], ['v3', 'v5'], ['v6', 'v0'], ['v2', 'b0']]
    _, next_nodes_to_be_added_names = qf.neighbours(circuit, inside_nodes + end_nodes, edges)
    result = [['v0', 'v2'],
              ['v2', 'v0']]
    assert next_nodes_to_be_added_names in result
