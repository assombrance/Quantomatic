import json
import sys
import numpy as np

diagram_file_path = sys.argv[1]
file = open(diagram_file_path)
content_string = file.read()
diagram_dictionary = json.loads(content_string)

if 'wire_vertices' in diagram_dictionary:
    wire_vertices_dictionary = diagram_dictionary['wire_vertices']
else:
    wire_vertices_dictionary = []
if 'undir_edges' in diagram_dictionary:
    undir_edges_dictionary = diagram_dictionary['undir_edges']
else:
    undir_edges_dictionary = []
all_numbered = True
inputs_order_dic = {}
outputs_order_dic = {}

for node_name in wire_vertices_dictionary:
    node_found_and_numbered = False
    for edge_name in undir_edges_dictionary:
        if dict(undir_edges_dictionary[edge_name])['tgt'] == node_name or \
                dict(undir_edges_dictionary[edge_name])['src'] == node_name:
            if 'data' in undir_edges_dictionary[edge_name] and \
                    'label' in dict(undir_edges_dictionary[edge_name])['data']:
                if 'input' in dict(undir_edges_dictionary[edge_name])['data']['label']:
                    inputs_order_dic[node_name] = dict(undir_edges_dictionary[edge_name])['data']['label']
                    node_found_and_numbered = True
                elif 'output' in dict(undir_edges_dictionary[edge_name])['data']['label']:
                    outputs_order_dic[node_name] = dict(undir_edges_dictionary[edge_name])['data']['label']
                    node_found_and_numbered = True
            else:
                all_numbered = False
    if not node_found_and_numbered:
        all_numbered = False

inputs_order_list = []
outputs_order_list = []

if all_numbered:
    for i in np.arange(len(inputs_order_dic)):
        node_found = False
        for node_name in inputs_order_dic:
            if inputs_order_dic[node_name] == "input" + str(i):
                inputs_order_list.append(node_name)
                node_found = True
        if not node_found:
            all_numbered = False
            break
    for i in np.arange(len(outputs_order_dic)):
        node_found = False
        for node_name in outputs_order_dic:
            if outputs_order_dic[node_name] == "output" + str(i):
                outputs_order_list.append(node_name)
                node_found = True
        if not node_found:
            all_numbered = False
            break

print(all_numbered, ";", inputs_order_list, ";", outputs_order_list)
