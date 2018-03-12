import json
import numpy as np
import math

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

    for i in np.arange(len(sorted_node_list) - 1):
        for node_name1 in sorted_node_list[i]:
            for node_name2 in sorted_node_list[i + 1]:
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
    for i in np.arange(len(sorted_node_list) - 1):
        for node_name1 in sorted_node_list[i]:
            for node_name2 in sorted_node_list[i + 1]:
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
    _, next_nodes_to_be_added_names, _ = qf.neighbours(circuit, inside_nodes + end_nodes, edges)
    result = [['v0', 'v2'],
              ['v2', 'v0']]
    assert next_nodes_to_be_added_names in result


def test_node_matrix_z():
    node = {'v6': {
        'edge_out': [['v2', 'v6'], ['v0', 'v6']],
        'edge_in': [['v6', 'b1']],
        'data': {'type': 'Z', 'value': '1'}}}
    result = np.linalg.norm(qf.nodes_matrix([node])[0] - np.matrix([[1, 0], [0, 0], [0, 0], [0, -1]]))
    assert result < 10**(-10)


def test_node_matrix_x():
    node = {'v6': {
        'edge_out': [['v2', 'v6'], ['v0', 'v6']],
        'edge_in': [['v6', 'b1']],
        'data': {'type': 'X', 'value': '1'}}}
    result = np.linalg.norm(qf.nodes_matrix([node])[0] - np.matrix([[0, 1], [1, 0], [1, 0], [0, 1]])/math.sqrt(2))
    assert result < 10**(-10)


def test_permutation_matrix():
    start = [['v0', 'v1'], ['v2', 'v1'], ['v0', 'b0']]
    end = [['v0', 'v1'], ['v0', 'b0'], ['v2', 'v1']]
    permutation = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    expected_result = qf.tensor_product(np.identity(2, dtype=complex), permutation)
    actual_result = qf.permutation_matrix(start, end)
    assert not (actual_result - expected_result).any()


def test_binary_dictionary_to_int():
    dictionary = {0: 1, 1: 1, 2: 0}
    assert qf.binary_dictionary_to_int(dictionary) == 3


def test_image_by_permutation():
    dictionary = {0: 1, 1: 2, 2: 0}
    assert qf.image_by_permutation(dictionary, 5) == 6


def test_build_permutation_dictionary():
    pre = [1, 2, 3]
    post = [2, 1, 3]
    # careful : binary representations are big-endian, so expected result may be a bit counter intuitive
    expected_result = {0: 0, 1: 2, 2: 1}
    actual_result = qf.build_permutation_dictionary(pre, post)
    assert expected_result == actual_result


def test_choose_root():
    inside_nodes = [{'v5': {'annotation': {'coord': [0.5, -1.25]}}},
                    {'v3': {'annotation': {'coord': [0.5, 0.0]}, 'data': {'type': 'hadamard', 'value': ''}}},
                    {'v2': {'annotation': {'coord': [0.5, 2.5]}, 'data': {'type': 'X', 'value': ''}}},
                    {'v9': {'annotation': {'coord': [2.0, 1.25]}, 'data': {'type': 'Z', 'value': '1/2'}}},
                    {'v6': {'annotation': {'coord': [2.0, 3.5]}, 'data': {'type': 'Z', 'value': '1'}}},
                    {'v1': {'annotation': {'coord': [2.0, 0.0]}, 'data': {'type': 'X', 'value': ''}}},
                    {'v0': {'annotation': {'coord': [0.5, 1.25]}, 'data': {'type': 'X', 'value': '3/4'}}}]
    connected_graphs = []
    root = qf.choose_root(inside_nodes, connected_graphs)
    for root_name in root:
        continue
    assert 'v' in root_name


def test_augment_graph():
    connected_graph = {'start_nodes': [],
                       'end_nodes': [],
                       'inside_nodes': [{'v2': {}}],
                       'edges': []}
    start_nodes = [{'b0': {}}, {'b1': {}}]
    end_nodes = [{'b3': {}}, {'b2': {}}]
    inside_nodes = [{'v2': {}}, {'v4': {}}, {'v1': {}}, {'v3': {}}, {'v0': {}}]
    edges = [['v3', 'b3'], ['v1', 'b1'], ['v0', 'v2'], ['v2', 'b2'],
             ['b0', 'v0'], ['v1', 'v0'], ['v1', 'v2'], ['v3', 'v4']]
    augmented_graph, _ = qf.augment_graph(connected_graph, start_nodes, end_nodes, inside_nodes, edges)
    print(augmented_graph)
    expected_augmented_graph = {'start_nodes': [],
                                'end_nodes': [{'b2': {}}],
                                'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                                'edges': [['v2', 'b2'], ['v0', 'v2'], ['v1', 'v2'], ['v1', 'v0']]}
    assert not qf.symmetric_difference(expected_augmented_graph['start_nodes'], augmented_graph['start_nodes'])
    assert not qf.symmetric_difference(expected_augmented_graph['end_nodes'], augmented_graph['end_nodes'])
    assert not qf.symmetric_difference(expected_augmented_graph['inside_nodes'], augmented_graph['inside_nodes'])
    assert not qf.symmetric_difference(expected_augmented_graph['edges'], augmented_graph['edges'])


def test_build_connected_graph():
    root = {'v2': {}}
    start_nodes = [{'b0': {}}, {'b1': {}}]
    end_nodes = [{'b3': {}}, {'b2': {}}]
    inside_nodes = [{'v2': {}}, {'v4': {}}, {'v1': {}}, {'v3': {}}, {'v0': {}}]
    edges = [['v3', 'b3'], ['v1', 'b1'], ['v0', 'v2'], ['v2', 'b2'],
             ['b0', 'v0'], ['v1', 'v0'], ['v1', 'v2'], ['v3', 'v4']]
    connected_graph = qf.build_connected_graph(root, start_nodes, end_nodes, inside_nodes, edges)
    expected_connected_graph = {'start_nodes': [{'b0': {}}, {'b1': {}}],
                                'end_nodes': [{'b2': {}}],
                                'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                                'edges': [['v2', 'b2'], ['v0', 'v2'], ['v1', 'v2'],
                                          ['v1', 'v0'], ['b0', 'v0'], ['v1', 'b1']]}
    assert not qf.symmetric_difference(expected_connected_graph['start_nodes'], connected_graph['start_nodes'])
    assert not qf.symmetric_difference(expected_connected_graph['end_nodes'], connected_graph['end_nodes'])
    assert not qf.symmetric_difference(expected_connected_graph['inside_nodes'], connected_graph['inside_nodes'])
    assert not qf.symmetric_difference(expected_connected_graph['edges'], connected_graph['edges'])


def test_end_detection_connected_graphs_true():
    connected_graphs = [{'start_nodes': [{'b0': {}}, {'b1': {}}],
                         'end_nodes': [{'b2': {}}],
                         'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                         'edges': [['v2', 'b2'], ['v0', 'v2'], ['v1', 'v2'],['v1', 'v0'], ['b0', 'v0'], ['v1', 'b1']]},
                        {'start_nodes': [],
                         'end_nodes': [{'b3': {}}],
                         'inside_nodes': [{'v3': {}}, {'v4': {}}],
                         'edges': [['v3', 'v4'], ['v3', 'b3']]}]
    start_nodes = [{'b0': {}}, {'b1': {}}]
    end_nodes = [{'b3': {}}, {'b2': {}}]
    inside_nodes = [{'v2': {}}, {'v4': {}}, {'v1': {}}, {'v3': {}}, {'v0': {}}]
    assert qf.end_detection_connected_graphs(connected_graphs, start_nodes, end_nodes, inside_nodes)


def test_end_detection_connected_graphs_false():
    # 'v1' is missing in the first graph
    connected_graphs = [{'start_nodes': [{'b0': {}}, {'b1': {}}],
                         'end_nodes': [{'b2': {}}],
                         'inside_nodes': [{'v2': {}}, {'v0': {}}],
                         'edges': [['v2', 'b2'], ['v0', 'v2'], ['v1', 'v2'],['v1', 'v0'], ['b0', 'v0'], ['v1', 'b1']]},
                        {'start_nodes': [],
                         'end_nodes': [{'b3': {}}],
                         'inside_nodes': [{'v3': {}}, {'v4': {}}],
                         'edges': [['v3', 'v4'], ['v3', 'b3']]}]
    start_nodes = [{'b0': {}}, {'b1': {}}]
    end_nodes = [{'b3': {}}, {'b2': {}}]
    inside_nodes = [{'v2': {}}, {'v4': {}}, {'v1': {}}, {'v3': {}}, {'v0': {}}]
    assert not qf.end_detection_connected_graphs(connected_graphs, start_nodes, end_nodes, inside_nodes)


def test_split_in_connected_graphs():
    start_nodes = [{'b0': {}}, {'b1': {}}]
    end_nodes = [{'b3': {}}, {'b2': {}}]
    inside_nodes = [{'v2': {}}, {'v4': {}}, {'v1': {}}, {'v3': {}}, {'v0': {}}]
    edges = [['v3', 'b3'], ['v1', 'b1'], ['v0', 'v2'], ['v2', 'b2'],
             ['b0', 'v0'], ['v1', 'v0'], ['v1', 'v2'], ['v3', 'v4']]
    connected_graphs = qf.split_in_connected_graphs(start_nodes, end_nodes, inside_nodes, edges)
    expected_connected_graphs = [{'start_nodes': [{'b0': {}}, {'b1': {}}],
                                  'end_nodes': [{'b2': {}}],
                                  'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                                  'edges': [['v2', 'b2'], ['v0', 'v2'], ['v1', 'v2'],
                                            ['v1', 'v0'], ['b0', 'v0'], ['v1', 'b1']]},
                                 {'start_nodes': [],
                                  'end_nodes': [{'b3': {}}],
                                  'inside_nodes': [{'v3': {}}, {'v4': {}}],
                                  'edges': [['v3', 'v4'], ['v3', 'b3']]}]
    assert not qf.symmetric_difference(expected_connected_graphs[0]['start_nodes'], connected_graphs[0]['start_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[0]['end_nodes'], connected_graphs[0]['end_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[0]['inside_nodes'], connected_graphs[0]['inside_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[0]['edges'], connected_graphs[0]['edges'])

    assert not qf.symmetric_difference(expected_connected_graphs[1]['start_nodes'], connected_graphs[1]['start_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[1]['end_nodes'], connected_graphs[1]['end_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[1]['inside_nodes'], connected_graphs[1]['inside_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[1]['edges'], connected_graphs[1]['edges'])
