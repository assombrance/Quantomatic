# coding=UTF-8
"""
Module focused on setting up the data to feed it to q_functions
"""
import json

from typing import List

import q_functions as qf
from data import Edge, Wire, Node, GenericMatrix, Graph


def main(diagram_file_path: str, inputs: List[str], outputs: List[str]) -> (bool, List[str], List[str], GenericMatrix):
    """main(diagram_file_path: str, inputs: List[str], outputs: List[str]) -> bool, List[str], List[str], GenericMatrix

    Computes the diagram matrix from the diagram given, the diagram inputs are precises separately (once integrated
    in Quantomatic, the inputs are asked to the user via a dialogue box)

    Args:
        diagram_file_path (path): Path the de diagram data file (extension : .qgraph, format : json)
        inputs (list[string]): Ordered list of inputs taken for this matrix calculation.
        outputs (list[string]): Ordered list of outputs taken for this matrix calculation.
    Returns:
         matrix: The matrix corresponding to the given diagram
    """
    wne_dir = load_dictionaries(diagram_file_path)
    wires, nodes, edges = dictionary_to_data(*wne_dir)  # the star is used to expand the tuple (unpack the argument)

    assumed_order, inputs, outputs = manage_i_o(wires, edges, inputs, outputs)

    start_wires = i_o_to_data(inputs, wires)
    end_wires = i_o_to_data(outputs, wires)

    m = qf.split_and_reunite(Graph(nodes, edges, start_wires, end_wires))
    return assumed_order, inputs, outputs, m


def check_for_doubles(inputs: list, outputs: list):
    """check_for_doubles(inputs: list, outputs: list)

    Raise an error if an element is both in the *inputs* and the *outputs*

    Args:
        inputs (list): diagram inputs
        outputs (list): diagram outputs

    Returns:
        None
    """
    in_out = inputs + outputs
    difference = qf.symmetric_difference(inputs, outputs)
    double = qf.symmetric_difference(in_out, difference)
    double_simplified = []
    i = 0
    for wire in double:
        i += 1
        if i % 2:
            double_simplified.append(wire)
    if double_simplified:
        raise ValueError('Common node in Inputs and Outputs : ' + str(double_simplified))


def load_dictionaries(diagram_file_path: str) -> (dict, dict, dict):
    """load_dictionaries(diagram_file_path: str) -> dict, dict, dict

    Loads the three dictionaries (wires, nodes and edges) from the file containing the graph

    Args:
        diagram_file_path (str): absolute or relative path to the *.qgraph* file

    Returns:
        dict, dict, dict: dictionaries for the wires, nodes and edges
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
    return wire_vertices_dictionary, node_vertices_dictionary, undir_edges_dictionary


def interpret_i_o(raw_io: List[str]) -> (List[str], List[str]):
    """interpret_i_o(raw_io: List[str]) -> List[str], List[str]

    Takes in the args from the main command and returns a list of inputs name and a list of outputs name

    Args:
        raw_io (List[str]): arg list from the command invoking main.py

    Returns:
        List[str], List[str]: list of inputs name and outputs name
    """
    if len(raw_io) < 2:
        raise ValueError("Bad argument list")
    raw_inputs = ""
    raw_outputs = ""
    if len(raw_io) == 3:
        raw_inputs = raw_io[2]
    if len(raw_io) > 3:
        raw_inputs = raw_io[2]
        raw_outputs = raw_io[3]
    _inputList = list(map(str, raw_inputs.strip('[]').strip(' ').strip('\n').split(',')))
    if '' in _inputList:
        _inputList.remove('')

    _outputList = list(map(str, raw_outputs.strip('[]').strip(' ').strip('\n').split(',')))
    if '' in _outputList:
        _outputList.remove('')
    return _inputList, _outputList


def i_o_to_data(i_o_list: List[str], wires: List[Wire]) -> List[Wire]:
    """i_o_to_data(i_o_list: List[str], wires: List[Wire]) -> List[Wire]

    Used to create the ordered list of input and output wires from the input and output names as well as the wires.

    Args:
        i_o_list (List[str]): ordered list of wire names
        wires (List[Wire]): list of all the wires in the graph

    Returns:
        List[Wire]: list of wire ordered
    """
    i_o_wires = []
    for i_o_name in i_o_list:
        found = False
        for wire in wires:
            if wire.name == i_o_name:
                i_o_wires.append(wire)
                found = True
                break
        if not found:
            raise ValueError('Boundary \'' + i_o_name + '\' not in wire_vertices dictionary')
    return i_o_wires


def manage_i_o(wires: List[Wire], edges: List[Edge], inputs_order_list: List[str],
               outputs_order_list: List[str]) -> (bool, List[str], List[str]):
    """manage_i_o(wires: List[Wire], edges: List[Edge], inputs_order_list: List[str], outputs_order_list: List[str]) -> bool, List[str], List[str]

    Looks for unlisted inputs and outputs, in case some are found, if listed as input or output in the file, add them
    otherwise, raise an error.

    Args:
        wires (List[Wire]): list of the wires in the diagram
        edges (List[Edge]): list of the edges in the diagram
        inputs_order_list (list[string]): user input list
        outputs_order_list (list[string]): user output list

    Returns:
        bool, List[str], List[str]: did the I/O change ? updated list of inputs and outputs
    """
    assumed_order = False
    for wire in wires:
        if wire.name not in inputs_order_list + outputs_order_list:
            for edge in edges:
                if edge.n1 == wire or edge.n2 == wire:
                    if 'input' in edge.label or 'in' in edge.label or 'i' in edge.label:
                        inputs_order_list.append(wire.name)
                        assumed_order = True
                    elif 'output' in edge.label or 'out' in edge.label or 'o' in edge.label:
                        outputs_order_list.append(wire.name)
                        assumed_order = True
                    else:
                        raise ValueError('Not all wire edges in Inputs + Outputs (missing \'' + wire.name + '\')')
    return assumed_order, inputs_order_list, outputs_order_list


def dictionary_to_data(wire_vertices_dictionary: dict, node_vertices_dictionary: dict,
                       undir_edges_dictionary: dict) -> (List[Wire], List[Node], List[Edge]):
    """dictionary_to_data(wire_vertices_dictionary: dict, node_vertices_dictionary: dict, undir_edges_dictionary: dict) -> List[Wire], List[Node], List[Edge]

    Converts the three dictionaries (wires, nodes and edges) to lists of classes defined in :ref:`data`

    Args:
        wire_vertices_dictionary (dict): dictionary containing the data about the wires
        node_vertices_dictionary (dict): dictionary containing the data about the nodes
        undir_edges_dictionary (dict): dictionary containing the data about the edges

    Returns:
        List[Wire], List[Node], List[Edge]: lists of wires, nodes and edges
    """
    wires = []  # type: List[Wire]
    for wire_name in wire_vertices_dictionary:
        wires.append(Wire(wire_name))

    nodes = []  # type: List[Node]
    for node_name in node_vertices_dictionary:
        angle = 0.
        node_type = 'Z'
        if 'data' in node_vertices_dictionary[node_name]:
            node_type = node_vertices_dictionary[node_name]['data']['type']
            angle = node_vertices_dictionary[node_name]['data']['value']
            if angle == '':
                angle = 0.
            else:
                angle = angle.replace('\\pi', '1')
                angle = angle.replace('pi', '1')
                angle = angle.replace('Pi', '1')
                angle = float(eval(angle))
        nodes.append(Node(node_name, angle, node_type))

    edges = []  # type: List[Edge]
    for edge_name in undir_edges_dictionary:
        label = ""
        if 'data' in undir_edges_dictionary[edge_name]:
            label = undir_edges_dictionary[edge_name]['data']['label']
        n1 = None
        n2 = None
        for node in nodes:
            if node.name == undir_edges_dictionary[edge_name]['src']:
                node.arity += 1
                n1 = node
            if node.name == undir_edges_dictionary[edge_name]['tgt']:
                node.arity += 1
                n2 = node
        for wire in wires:
            if wire.name == undir_edges_dictionary[edge_name]['src']:
                n1 = wire
            if wire.name == undir_edges_dictionary[edge_name]['tgt']:
                n2 = wire
        edges.append(Edge(edge_name, n1, n2, label))

    return wires, nodes, edges
