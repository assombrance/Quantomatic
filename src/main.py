import json

import q_functions as qf


def main(diagram_file_path, inputs_order_list, outputs_order_list):
    """Computes the diagram matrix from the diagram given, the diagram inputs are precises separately (once integrated
    in Quantomatic, the inputs are asked to the user via a dialogue box)

    Args:
          diagram_file_path (path): Path the de diagram data file (extension : .qgraph, format : json)
          inputs_order_list (list[end_nodes_names]): The list on the inputs taken for this matrix. Other end nodes are outputs
    Returns:
         matrix: The matrix corresponding to the given diagram
    """
    file = open(diagram_file_path)
    content_string = file.read()
    diagram_dictionary = json.loads(content_string)
    if 'wire_vertices' in diagram_dictionary:
        wire_vertices_dictionary = diagram_dictionary['wire_vertices']
    else:
        wire_vertices_dictionary = []
    if 'node_vertices' in diagram_dictionary:
        node_vertices_dictionary = diagram_dictionary['node_vertices']
    else:
        node_vertices_dictionary = []
    if 'undir_edges' in diagram_dictionary:
        undir_edges_dictionary = diagram_dictionary['undir_edges']
    else:
        undir_edges_dictionary = []
    for node_name in wire_vertices_dictionary:
        if node_name not in inputs_order_list + outputs_order_list:
            raise NameError('Not all wire edges in Inputs + Outputs')
    start_nodes = []
    for node_name in inputs_order_list:
        start_nodes.append({node_name: wire_vertices_dictionary[node_name]})
    end_nodes = []
    for node_name in outputs_order_list:
        end_nodes.append({node_name: wire_vertices_dictionary[node_name]})
    inside_nodes = []
    for node in node_vertices_dictionary:
        inside_nodes.append({node: node_vertices_dictionary[node]})
    edges_list = []
    for edge_dictionary in undir_edges_dictionary:
        edge = []
        for node in undir_edges_dictionary[edge_dictionary]:
            edge.append(undir_edges_dictionary[edge_dictionary][node])
        edges_list.append(edge)
    return qf.main_algo(start_nodes, end_nodes, inside_nodes, edges_list)


def debug(diagram_file_path):
    """ Prints in a semi developed view the file loaded

    Args:
        diagram_file_path (path): file to print path
    Returns:
         void
    """
    file = open(diagram_file_path)
    content_string = file.read()
    diagram_dictionary = json.loads(content_string)
    for entry in diagram_dictionary:
        print(entry, ': ')
        for entry2 in diagram_dictionary[entry]:
            print('    ', entry2, ': ', diagram_dictionary[entry][entry2])


# debug('D:/Users/Henri/Documents/Esisar/5A/PFE/Info quantique/Quantomatic/zx-project-1.1/graphs/sample.qgraph')
path_to_project = 'D:/Users/Henri/Documents/Esisar/5A/PFE/Info quantique/Quantomatic/zx-project_scalar_rules/graphs/'
# print(main(path_to_project + 'test.qgraph', ['b0', 'b1'], ['b2', 'b3']))
print(main(path_to_project + 'test2.qgraph', [], []))
print(main(path_to_project + 'test3.qgraph', [], []))
