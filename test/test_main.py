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
    edges = [{'e0': ['v0', 'v2']}, {'e1': ['v6', 'v2']}, {'e2': ['b1', 'v6']}, {'e4': ['v5', 'b2']},
             {'e5': ['v3', 'v0']}, {'e6': ['v1', 'v9']}, {'e7': ['v1', 'b3']}, {'e8': ['v9', 'v2']},
             {'e5': ['v1', 'v5']}, {'e10': ['v3', 'v5']}, {'e11': ['v6', 'v0']}, {'e12': ['v2', 'b0']}]
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
    edges = [{'e0': ['v0', 'v2']}, {'e1': ['v6', 'v2']}, {'e2': ['b1', 'v6']}, {'e4': ['v5', 'b2']},
             {'e5': ['v3', 'v0']}, {'e6': ['v1', 'v9']}, {'e7': ['v1', 'b3']}, {'e8': ['v9', 'v2']},
             {'e5': ['v1', 'v5']}, {'e10': ['v3', 'v5']}, {'e11': ['v6', 'v0']}, {'e12': ['v2', 'b0']}]
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
    # careful : binary representations are little-endian, so expected result may be a bit counter intuitive:
    # 10 is stored as [0, 1, 0, 1] and not the opposite
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
    edges = [{'e0': ['v3', 'b3']}, {'e1': ['v1', 'b1']}, {'e2': ['v0', 'v2']}, {'e3': ['v2', 'b2']},
             {'e4': ['b0', 'v0']}, {'e5': ['v1', 'v0']}, {'e6': ['v1', 'v2']}, {'e7': ['v3', 'v4']}]
    augmented_graph, _ = qf.augment_graph(connected_graph, start_nodes, end_nodes, inside_nodes, edges)
    # print(augmented_graph)
    expected_augmented_graph = {'start_nodes': [],
                                'end_nodes': [{'b2': {}}],
                                'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                                'edges': [{'e3': ['v2', 'b2']}, {'e2': ['v0', 'v2']},
                                          {'e6': ['v1', 'v2']}, {'e5': ['v1', 'v0']}]}
    assert not qf.symmetric_difference(expected_augmented_graph['start_nodes'], augmented_graph['start_nodes'])
    assert not qf.symmetric_difference(expected_augmented_graph['end_nodes'], augmented_graph['end_nodes'])
    assert not qf.symmetric_difference(expected_augmented_graph['inside_nodes'], augmented_graph['inside_nodes'])
    assert not qf.symmetric_difference(expected_augmented_graph['edges'], augmented_graph['edges'])


def test_build_connected_graph():
    root = {'v2': {}}
    start_nodes = [{'b0': {}}, {'b1': {}}]
    end_nodes = [{'b3': {}}, {'b2': {}}]
    inside_nodes = [{'v2': {}}, {'v4': {}}, {'v1': {}}, {'v3': {}}, {'v0': {}}]
    edges = [{'e0': ['v3', 'b3']}, {'e1': ['v1', 'b1']}, {'e2': ['v0', 'v2']}, {'e3': ['v2', 'b2']},
             {'e4': ['b0', 'v0']}, {'e5': ['v1', 'v0']}, {'e6': ['v1', 'v2']}, {'e7': ['v3', 'v4']}]
    connected_graph = qf.build_connected_graph(root, start_nodes, end_nodes, inside_nodes, edges)
    expected_connected_graph = {'start_nodes': [{'b0': {}}, {'b1': {}}],
                                'end_nodes': [{'b2': {}}],
                                'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                                'edges': [{'e3': ['v2', 'b2']}, {'e2': ['v0', 'v2']}, {'e6': ['v1', 'v2']},
                                          {'e5': ['v1', 'v0']}, {'e4': ['b0', 'v0']}, {'e1': ['v1', 'b1']}]}
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
    edges = [{'e0': ['v3', 'b3']}, {'e1': ['v1', 'b1']}, {'e2': ['v0', 'v2']}, {'e3': ['v2', 'b2']},
             {'e4': ['b0', 'v0']}, {'e5': ['v1', 'v0']}, {'e6': ['v1', 'v2']}, {'e7': ['v3', 'v4']}]
    connected_graphs = qf.split_in_connected_graphs(start_nodes, end_nodes, inside_nodes, edges)
    expected_connected_graphs = [{'start_nodes': [{'b0': {}}, {'b1': {}}],
                                  'end_nodes': [{'b2': {}}],
                                  'inside_nodes': [{'v2': {}}, {'v1': {}}, {'v0': {}}],
                                  'edges': [{'e3': ['v2', 'b2']}, {'e2': ['v0', 'v2']}, {'e6': ['v1', 'v2']},
                                            {'e5': ['v1', 'v0']}, {'e4': ['b0', 'v0']}, {'e1': ['v1', 'b1']}]},
                                 {'start_nodes': [],
                                  'end_nodes': [{'b3': {}}],
                                  'inside_nodes': [{'v3': {}}, {'v4': {}}],
                                  'edges': [{'e7': ['v3', 'v4']}, {'e0': ['v3', 'b3']}]}]
    assert not qf.symmetric_difference(expected_connected_graphs[0]['start_nodes'], connected_graphs[0]['start_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[0]['end_nodes'], connected_graphs[0]['end_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[0]['inside_nodes'], connected_graphs[0]['inside_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[0]['edges'], connected_graphs[0]['edges'])

    assert not qf.symmetric_difference(expected_connected_graphs[1]['start_nodes'], connected_graphs[1]['start_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[1]['end_nodes'], connected_graphs[1]['end_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[1]['inside_nodes'], connected_graphs[1]['inside_nodes'])
    assert not qf.symmetric_difference(expected_connected_graphs[1]['edges'], connected_graphs[1]['edges'])


def test_scalar_matrix_z_0():
    inside_nodes = [{'z0': {'data': {'type': 'Z', 'value': '0'}}}]
    edges = []
    scalar = qf.scalar_matrix(inside_nodes, edges)
    scalar = np.array(scalar)[0][0]
    assert scalar == 2


def test_scalar_matrix_x_0():
    inside_nodes = [{'x0': {'data': {'type': 'X', 'value': '0'}}}]
    edges = []
    scalar = qf.scalar_matrix(inside_nodes, edges)
    scalar = np.array(scalar)[0][0]
    assert scalar == 2


def test_nodes_list_to_nodes_dictionary():
    nodes_list = [{'b0': {'annotation': {'boundary': True, 'coord': [-6.5, 3.75]}}},
                  {'b1': {'annotation': {'boundary': True, 'coord': [-4.25, 3.75]}}},
                  {'b2': {'annotation': {'boundary': True, 'coord': [-5.5, -6.0]}}},
                  {'b3': {'annotation': {'boundary': True, 'coord': [0.5, -4.25]}}},
                  {'v1': {'annotation': {'coord': [-2.0, 2.75]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v3': {'annotation': {'coord': [-0.25, 0.5]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v4': {'annotation': {'coord': [1.25, 0.0]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v5': {'annotation': {'coord': [2.75, 1.5]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v0': {'annotation': {'coord': [-7.5, 2.5]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v2': {'annotation': {'coord': [-5.5, -0.5]}}}]
    nodes_dictionary = {'b3': {'annotation': {'boundary': True, 'coord': [0.5, -4.25]}},
                        'v1': {'annotation': {'coord': [-2.0, 2.75]}, 'data': {'type': 'X', 'value': ''}},
                        'v3': {'annotation': {'coord': [-0.25, 0.5]}, 'data': {'type': 'X', 'value': ''}},
                        'v4': {'annotation': {'coord': [1.25, 0.0]}, 'data': {'type': 'X', 'value': ''}},
                        'b1': {'annotation': {'boundary': True, 'coord': [-4.25, 3.75]}},
                        'b0': {'annotation': {'boundary': True, 'coord': [-6.5, 3.75]}},
                        'v0': {'annotation': {'coord': [-7.5, 2.5]}, 'data': {'type': 'X', 'value': ''}},
                        'v5': {'annotation': {'coord': [2.75, 1.5]}, 'data': {'type': 'X', 'value': ''}},
                        'b2': {'annotation': {'boundary': True, 'coord': [-5.5, -6.0]}},
                        'v2': {'annotation': {'coord': [-5.5, -0.5]}}}
    assert qf.nodes_list_to_nodes_dictionary(nodes_list) == nodes_dictionary


def test_nodes_dictionary_to_nodes_list():
    nodes_list = [{'b0': {'annotation': {'boundary': True, 'coord': [-6.5, 3.75]}}},
                  {'b1': {'annotation': {'boundary': True, 'coord': [-4.25, 3.75]}}},
                  {'b2': {'annotation': {'boundary': True, 'coord': [-5.5, -6.0]}}},
                  {'b3': {'annotation': {'boundary': True, 'coord': [0.5, -4.25]}}},
                  {'v1': {'annotation': {'coord': [-2.0, 2.75]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v3': {'annotation': {'coord': [-0.25, 0.5]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v4': {'annotation': {'coord': [1.25, 0.0]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v5': {'annotation': {'coord': [2.75, 1.5]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v0': {'annotation': {'coord': [-7.5, 2.5]}, 'data': {'type': 'X', 'value': ''}}},
                  {'v2': {'annotation': {'coord': [-5.5, -0.5]}}}]
    nodes_dictionary = {'b3': {'annotation': {'boundary': True, 'coord': [0.5, -4.25]}},
                        'v1': {'annotation': {'coord': [-2.0, 2.75]}, 'data': {'type': 'X', 'value': ''}},
                        'v3': {'annotation': {'coord': [-0.25, 0.5]}, 'data': {'type': 'X', 'value': ''}},
                        'v4': {'annotation': {'coord': [1.25, 0.0]}, 'data': {'type': 'X', 'value': ''}},
                        'b1': {'annotation': {'boundary': True, 'coord': [-4.25, 3.75]}},
                        'b0': {'annotation': {'boundary': True, 'coord': [-6.5, 3.75]}},
                        'v0': {'annotation': {'coord': [-7.5, 2.5]}, 'data': {'type': 'X', 'value': ''}},
                        'v5': {'annotation': {'coord': [2.75, 1.5]}, 'data': {'type': 'X', 'value': ''}},
                        'b2': {'annotation': {'boundary': True, 'coord': [-5.5, -6.0]}},
                        'v2': {'annotation': {'coord': [-5.5, -0.5]}}}
    assert qf.symmetric_difference(qf.nodes_dictionary_to_nodes_list(nodes_dictionary), nodes_list) == []


def test_tensor_power():
    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
    expected_result = np.matrix([[1, 1, 1, 1],
                                 [1, -1, 1, -1],
                                 [1, 1, -1, -1],
                                 [1, -1, -1, 1]]) / 2
    assert not (qf.tensor_power(h, 0) - [[1]]).any()
    assert not (qf.tensor_power(h, 1) - h).any()
    assert np.linalg.norm(qf.tensor_power(h, 2) - expected_result) < 10**(-10)


def test_tensor_product():
    m1 = np.identity(2)
    m2 = np.matrix([[0, 1],
                    [1, 0]])
    expected_result = np.matrix([[0, 1, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])
    assert not (qf.tensor_product(m1, m2) - expected_result).any()


def test_end_detection_main_algo_false():
    circuit_names = ['b3', 'v3']
    inside_nodes = [{'v3': {'edge_out': [['v3', 'v4']], 'data': {'type': 'X', 'value': ''},
                            'edge_in': [['v3', 'b3']], 'annotation': {'coord': [-0.25, 0.5]}}},
                    {'v4': {'data': {'type': 'X', 'value': ''}, 'annotation': {'coord': [1.25, 0.0]}}}]
    assert not qf.end_detection_main_algo(circuit_names, inside_nodes)


def test_end_detection_main_algo_true():
    circuit_names = ['b3', 'v3', 'v4']
    inside_nodes = [{'v3': {'edge_out': [['v3', 'v4']], 'data': {'type': 'X', 'value': ''},
                            'edge_in': [['v3', 'b3']], 'annotation': {'coord': [-0.25, 0.5]}}},
                    {'v4': {'data': {'type': 'X', 'value': ''}, 'annotation': {'coord': [1.25, 0.0]}}}]
    assert qf.end_detection_main_algo(circuit_names, inside_nodes)


def test_symmetric_difference():
    a = [1, 2, 3]
    b = [1, 3, 2]
    c = [1, 2]
    assert qf.symmetric_difference(a, b) == []
    assert qf.symmetric_difference(a, c) == [3]
    assert qf.symmetric_difference(c, a) == [3]


def test_remove_end_nodes_neighbours():
    next_nodes_to_be_added = [{'v1': {'annotation': {'coord': [2.5, -1.0]}, 'data': {'type': 'X', 'value': ''},
                                      'edge_out': [['b3', 'v1']], 'edge_in': [['v5', 'v1']]}},
                              {'b2': {'annotation': {'boundary': True, 'coord': [0.5, -2.75]}, 'edge_out': [],
                                      'edge_in': [['b2', 'v5']]}}]
    next_nodes_to_be_added_names = ['v1', 'b2']
    end_nodes_names = ['b2', 'b3']
    _, result = qf.remove_end_nodes_neighbours(next_nodes_to_be_added, next_nodes_to_be_added_names, end_nodes_names)
    assert result == ['v1']


def test_pre_permutation_edge_order_management():
    start_nodes_order = [{'v6': {'annotation': {'coord': [0.5, 3.75]}, 'data': {'value': '1', 'type': 'Z'}}}]
    edges = [{'e0': ['v0', 'v2']}, {'e1': ['v2', 'v6']}, {'e2': ['b1', 'v6']}, {'e4': ['v5', 'b2']},
             {'e5': ['v3', 'v0']}, {'e6': ['v1', 'v9']}, {'e7': ['v1', 'b3']}, {'e8': ['v9', 'v2']},
             {'e5': ['v1', 'v5']}, {'e10': ['v3', 'v5']}, {'e11': ['v0', 'v6']}, {'e12': ['b0', 'v2']}]
    circuit_names = ['b0', 'b1']
    next_nodes_to_be_added_names = ['v6']
    result = qf.pre_permutation_edge_order_management(start_nodes_order, edges,
                                                      circuit_names, next_nodes_to_be_added_names)
    # print(result)
    expected_start_edges_order = [{'e1': ['v2', 'v6']}, {'e11': ['v0', 'v6']}, {'e12': ['b0', 'v2']}]
    assert result == expected_start_edges_order


def test_post_permutation_edge_order_management():
    end_nodes_order = [{'v0': {'data': {'value': '3/4', 'type': 'X'}, 'annotation': {'coord': [0.5, 2.75]}}},
                       {'v9': {'data': {'value': '1/2', 'type': 'Z'}, 'annotation': {'coord': [2.5, 0.5]}}}]
    circuit_names = ['b0', 'b1', 'v2']
    edges = [{'e0': ['v0', 'v2']}, {'e1': ['v2', 'v6']}, {'e2': ['b1', 'v6']}, {'e4': ['v5', 'b2']},
             {'e5': ['v3', 'v0']}, {'e6': ['v1', 'v9']}, {'e7': ['v1', 'b3']}, {'e8': ['v9', 'v2']},
             {'e5': ['v1', 'v5']}, {'e10': ['v3', 'v5']}, {'e11': ['v0', 'v6']}, {'e12': ['b0', 'v2']}]
    next_nodes_to_be_added_names = ['v0', 'v9']
    end_edges_order, end_nodes_order = qf.post_permutation_edge_order_management(end_nodes_order, circuit_names, edges,
                                                                                 next_nodes_to_be_added_names)
    expected_end_edges_order = [{'e0': ['v0', 'v2']}, {'e8': ['v9', 'v2']}, {'e1': ['v2', 'v6']}, {'e2': ['b1', 'v6']}]
    expected_end_nodes_order = [{'v0': {'edge_out': [{'e5': ['v3', 'v0']}, {'e11': ['v0', 'v6']}],
                                        'data': {'value': '3/4', 'type': 'X'},
                                        'annotation': {'coord': [0.5, 2.75]}, 'edge_in': [{'e0': ['v0', 'v2']}]}},
                                {'v9': {'edge_out': [{'e6': ['v1', 'v9']}], 'data': {'value': '1/2', 'type': 'Z'},
                                        'annotation': {'coord': [2.5, 0.5]}, 'edge_in': [{'e8': ['v9', 'v2']}]}}]
    assert end_edges_order == expected_end_edges_order
    assert end_nodes_order == expected_end_nodes_order


def symmetric_difference_edges(a, b):
    result = []
    for i in a:
        found = False
        for j in b:
            if i[0] in j and i[1] in j:
                found = True
                break
        if not found:
            result.append(i)
    for i in b:
        found = False
        for j in a:
            if i[0] in j and i[1] in j:
                found = True
                break
        if not found:
            result.append(i)
    return result


def test_connected_graph_matrix():
    start_nodes = [{'b0': {'annotation': {'coord': [-6.5, 3.75], 'boundary': True}}},
                   {'b1': {'annotation': {'coord': [-4.25, 3.75], 'boundary': True}}}]
    end_nodes = [{'b2': {'annotation': {'coord': [-5.25, -5.0], 'boundary': True}}}]
    inside_nodes = [{'v0': {'data': {'type': 'X', 'value': ''}, 'annotation': {'coord': [-7.5, 2.5]}}},
                    {'v1': {'data': {'type': 'X', 'value': ''}, 'annotation': {'coord': [-2.0, 2.75]}}},
                    {'v2': {'annotation': {'coord': [-5.5, -0.5]}}}]
    edges = [{'e0': ['v0', 'b0']}, {'e1': ['v0', 'v1']}, {'e2': ['v2', 'v0']},
             {'e3': ['v2', 'v1']}, {'e4': ['b1', 'v1']}, {'e5': ['b2', 'v2']}]
    matrix = qf.connected_graph_matrix(start_nodes, end_nodes, inside_nodes, edges)
    expected_matrix = np.matrix([[0.5, 0, 0, 0.5],
                                 [0.5, 0, 0, 0.5]])
    assert np.linalg.norm(matrix - expected_matrix) < 10**(-10)


def test_main_algo():
    start_nodes = [{'b0': {'annotation': {'coord': [-6.5, 3.75], 'boundary': True}}},
                   {'b1': {'annotation': {'coord': [-4.25, 3.75], 'boundary': True}}}]
    end_nodes = [{'b2': {'annotation': {'coord': [-5.25, -5.0], 'boundary': True}}},
                 {'b3': {'annotation': {'coord': [-0.5, -5.0], 'boundary': True}}}]
    inside_nodes = [{'v1': {'data': {'value': '', 'type': 'X'}, 'annotation': {'coord': [-2.0, 2.75]}}},
                    {'v3': {'data': {'value': '', 'type': 'X'}, 'annotation': {'coord': [-0.25, 0.5]}}},
                    {'v2': {'annotation': {'coord': [-5.5, -0.5]}}},
                    {'v4': {'data': {'value': '', 'type': 'X'}, 'annotation': {'coord': [1.25, 0.0]}}},
                    {'v0': {'data': {'value': '', 'type': 'X'}, 'annotation': {'coord': [-7.5, 2.5]}}},
                    {'v5': {'data': {'value': '', 'type': 'X'}, 'annotation': {'coord': [2.75, 1.5]}}}]
    edges = [{'e5': ['b2', 'v2']}, {'e8': ['v0', 'v1']}, {'e1': ['v0', 'b0']}, {'e10': ['b3', 'v3']},
             {'e9': ['v4', 'v3']}, {'e7': ['v2', 'v1']}, {'e4': ['v2', 'v0']}, {'e6': ['b1', 'v1']}]
    matrix = qf.main_algo(start_nodes, end_nodes, inside_nodes, edges)
    expected_matrix = np.matrix([[1, 0, 0, 1],
                                 [0, 0, 0, 0],
                                 [1, 0, 0, 1],
                                 [0, 0, 0, 0]]) * math.sqrt(2)
    assert np.linalg.norm(matrix - expected_matrix) < 10**(-10)
