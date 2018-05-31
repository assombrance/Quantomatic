# encoding=UTF-8
"""
Preparation step, checks if I/O order can be deduced without user being required to input anything
"""
import sys
import numpy as np

import loader

if __name__ == '__main__':
    diagram_file_path = sys.argv[1]

    wne_dir = loader.load_dictionaries(diagram_file_path)
    wires, nodes, edges = loader.dictionary_to_data(*wne_dir)

    input_names = {"input", "in", "i"}
    output_names = {"output", "out", "o"}
    all_numbered = True
    inputs_order_dic = {}
    outputs_order_dic = {}

    for wire in wires:
        node_found_and_numbered = False
        for edge in edges:
            if wire in edge:
                if input_names.intersection(edge.label):
                    inputs_order_dic[wire.name] = edge.label
                    node_found_and_numbered = True
                elif output_names.intersection(edge.label):
                    outputs_order_dic[wire.name] = edge.label
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
            input_names_numbered = {input_name + str(i) for input_name in input_names}
            for node_name in inputs_order_dic:
                if inputs_order_dic[node_name] in input_names_numbered:
                    inputs_order_list.append(node_name)
                    node_found = True
            if not node_found:
                all_numbered = False
                break
        for i in np.arange(len(outputs_order_dic)):
            node_found = False
            output_names_numbered = {output_name + str(i) for output_name in output_names}
            for node_name in outputs_order_dic:
                if outputs_order_dic[node_name] in output_names_numbered:
                    outputs_order_list.append(node_name)
                    node_found = True
            if not node_found:
                all_numbered = False
                break

    print(all_numbered, ";", inputs_order_list, ";", outputs_order_list)
