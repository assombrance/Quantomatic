# coding=UTF-8
"""
Module focused on the fusion_matrices, taking two diagram's matrices, with those diagrams linked by edges, and computing
the resulting matrix.

This module contains non trivial calculations. If some of them are unclear, you can find documentation on the subject on
the GitHub project, in the
`"calculs-prealables.pdf" <https://github.com/assombrance/Quantomatic/blob/master/calculs-prealables.pdf>`_ document.
"""
from typing import List
from copy import deepcopy

import numpy as np

from data import GenericMatrix, ConnectionPoint, InterMatrixLink, EnhancedInt, UsedFragment


def fusion_matrices(m1: GenericMatrix, m2: GenericMatrix, inputs: List[ConnectionPoint],
                    outputs: List[ConnectionPoint], links: List[InterMatrixLink]) -> GenericMatrix:
    """ .. _fusion_matrices:

    Main component of the V2 of the algorithm uses clever algorithms to lower the complexity.
    The two matrices are in sequence, but don't have to be fully connected (or at all, in that case we fall back in the
    tensor product case)

    Args:
        m1 (GenericMatrix):
        m2 (GenericMatrix):
        inputs (List[ConnectionPoint]):
        outputs (List[ConnectionPoint]):
        links (List[InterMatrixLink]):

    Returns:
        GenericMatrix:
    """

    def reindex_connection_points(_removed_point: ConnectionPoint, _links: List[InterMatrixLink],
                                  _inputs: List[ConnectionPoint], _outputs: List[ConnectionPoint]):
        """Edits inplace the indexes of the ConnectionPoints if the *_removed_point* is on the same matrix, side and
        it's index was lower than the concerned ConnectionPoints.

        Args:
            _removed_point (ConnectionPoint):
            _links (List[InterMatrixLink]):
            _inputs (List[ConnectionPoint]):
            _outputs (List[ConnectionPoint]):
        """
        for _link in _links:
            if _link.point1.is_matrix_2 == _removed_point.is_matrix_2 and _link.point1.is_out == _removed_point.is_out \
                    and _link.point1.index > _removed_point.index:
                _link.point1.index -= 1
            if _link.point2.is_matrix_2 == _removed_point.is_matrix_2 and _link.point2.is_out == _removed_point.is_out \
                    and _link.point2.index > _removed_point.index:
                _link.point2.index -= 1
        for _input_point in _inputs:
            if _input_point.is_matrix_2 == _removed_point.is_matrix_2 and _input_point.is_out == _removed_point.is_out \
                    and _input_point.index > _removed_point.index:
                _input_point.index -= 1
        for _output_point in _outputs:
            if _output_point.is_matrix_2 == _removed_point.is_matrix_2 \
                    and _output_point.is_out == _removed_point.is_out \
                    and _output_point.index > _removed_point.index:
                _output_point.index -= 1

    # use of while instead of for to be able to remove values inplace
    iteration_index = 0
    while iteration_index < len(links):
        removed_points = []  # type: List[ConnectionPoint]
        link = links[iteration_index]
        # we don't need to memorize the points we'll add because we will always add them last
        if link.point1.is_matrix_2 == link.point2.is_matrix_2:
            if link.point1.is_out != link.point2.is_out:
                # set the point2 to the same side of the matrix as point1
                # and remove the resulting cap or cup
                if link.point1.is_matrix_2:
                    if link.point1.is_out:
                        m2 = input_to_output(m2, link.point1.index)
                        m2 = remove_cap(m2, link.point2.index, -1)
                    else:
                        m2 = input_to_output(m2, link.point2.index)
                        m2 = remove_cap(m2, link.point1.index, -1)
                else:
                    if link.point1.is_out:
                        m1 = input_to_output(m1, link.point1.index)
                        m1 = remove_cap(m1, link.point2.index, -1)
                    else:
                        m1 = input_to_output(m1, link.point2.index)
                        m1 = remove_cap(m1, link.point1.index, -1)
            elif link.point1.is_out:
                # remove cup
                if link.point1.is_matrix_2:
                    m2 = remove_cup(m2, link.point1.index, link.point2.index)
                else:
                    m1 = remove_cup(m1, link.point1.index, link.point2.index)
            else:
                # remove cap
                if link.point1.is_matrix_2:
                    m2 = remove_cap(m2, link.point1.index, link.point2.index)
                else:
                    m1 = remove_cap(m1, link.point1.index, link.point2.index)
            removed_points.append(link.point1)
            removed_points.append(link.point2)
        elif not link.point1.is_matrix_2 and not link.point1.is_out:
            # input from matrix 1 connected to matrix 2
            # switch point1 to be an output
            m1 = input_to_output(m1, link.point1.index)
            removed_points.append(deepcopy(link.point1))
            link.point1.is_out = True
            m1_height = EnhancedInt(m1.shape[0])
            link.point1.index = max((m1_height - 1).bit_length() - 1, 0)
        elif not link.point2.is_matrix_2 and not link.point2.is_out:
            # input from matrix 1 connected to matrix 2
            # switch point2 to be an output
            m1 = input_to_output(m1, link.point2.index)
            removed_points.append(deepcopy(link.point2))
            link.point2.is_out = True
            m1_height = EnhancedInt(m1.shape[0])
            link.point2.index = max((m1_height - 1).bit_length() - 1, 0)
        elif link.point2.is_matrix_2 and link.point2.is_out:
            # output from matrix 2 connected to matrix 1
            # switch point2 to be an input
            m2 = output_to_input(m2, link.point2.index)
            removed_points.append(deepcopy(link.point2))
            link.point2.is_out = False
            m2_width = EnhancedInt(m2.shape[1])
            link.point2.index = max((m2_width - 1).bit_length() - 1, 0)
        elif link.point1.is_matrix_2 and link.point1.is_out:
            # output from matrix 2 connected to matrix 1
            # switch point1 to be an input
            m2 = output_to_input(m2, link.point1.index)
            removed_points.append(deepcopy(link.point1))
            link.point1.is_out = False
            m2_width = m2.shape[1]
            link.point1.index = max((m2_width - 1).bit_length() - 1, 0)
        # move the indexes of the other points to the correct new positions
        if link.point1 in removed_points and link.point2 in removed_points:
            del links[iteration_index]
        else:
            iteration_index += 1
        if link.point1.index == -1:
            if not link.point1.is_matrix_2:
                if not link.point1.is_out:
                    link.point1.index = max(EnhancedInt(m1.shape[1] - 1).bit_length() - 1, 0)
                else:
                    link.point1.index = max(EnhancedInt(m1.shape[0] - 1).bit_length() - 1, 0)
            else:
                if not link.point1.is_out:
                    link.point1.index = max(EnhancedInt(m2.shape[1] - 1).bit_length() - 1, 0)
                else:
                    link.point1.index = max(EnhancedInt(m2.shape[0] - 1).bit_length() - 1, 0)
        for removed_point in removed_points:
            reindex_connection_points(removed_point, links, inputs, outputs)
    for input_point in inputs:
        if input_point.is_out:
            # change this input to be an input for the matrix
            if not input_point.is_matrix_2:
                m1 = output_to_input(m1, input_point.index)
                removed_point = deepcopy(input_point)
                input_point.index = EnhancedInt.from_list(m1.shape[1]).bit_length() - 1
            else:
                m2 = output_to_input(m2, input_point.index)
                removed_point = deepcopy(input_point)
                input_point.index = EnhancedInt.from_list(m2.shape[1]).bit_length() - 1
            input_point.is_out = False
            reindex_connection_points(removed_point, links, inputs, outputs)
    for output_point in outputs:
        if not output_point.is_out:
            # change this output to be an output for the matrix
            if not output_point.is_matrix_2:
                m1 = input_to_output(m1, output_point.index)
                removed_point = deepcopy(output_point)
                output_point.index = EnhancedInt.from_list(m1.shape[0]).bit_length() - 1
            else:
                m2 = input_to_output(m2, output_point.index)
                removed_point = deepcopy(output_point)
                output_point.index = EnhancedInt.from_list(m2.shape[0]).bit_length() - 1
            output_point.is_out = True
            reindex_connection_points(removed_point, links, inputs, outputs)

    # prepare lists to reorder m1
    m1_m2_links = [link for link in links if link.point1.is_matrix_2 != link.point2.is_matrix_2]
    m1_m2_links_m1_sorted = sorted(m1_m2_links, key=lambda _link: _link.point1.index if not _link.point1.is_matrix_2
                                   else _link.point2.index)
    covering = len(m1_m2_links)

    m1_inputs = [input_point for input_point in inputs if not input_point.is_matrix_2]
    m1_inputs_sorted = sorted(m1_inputs, key=lambda input_m1: input_m1.index)

    m1_outputs_to_m2 = [link.point2 if link.point1.is_matrix_2 else link.point1 for link in m1_m2_links_m1_sorted]
    m1_outputs_to_output = [output_point for output_point in outputs if not output_point.is_matrix_2]

    m1_outputs = m1_outputs_to_output + m1_outputs_to_m2
    m1_outputs_sorted = sorted(m1_outputs, key=lambda output_m1: output_m1.index)

    # order m1
    m1_ordered = order_matrix(m1, m1_inputs, m1_inputs_sorted, m1_outputs_sorted, m1_outputs)

    # prepare lists to reorder m2
    m2_inputs_from_inputs = [input_point for input_point in inputs if input_point.is_matrix_2]
    m2_inputs_from_m1 = [link.point1 if link.point1.is_matrix_2 else link.point2 for link in m1_m2_links_m1_sorted]

    m2_inputs = m2_inputs_from_m1 + m2_inputs_from_inputs
    m2_inputs_sorted = sorted(m2_inputs, key=lambda input_m2: input_m2.index)

    m2_outputs = [output_point for output_point in outputs if output_point.is_matrix_2]
    m2_outputs_sorted = sorted(m2_outputs, key=lambda output_m2: output_m2.index)

    # order m2
    m2_ordered = order_matrix(m2, m2_inputs, m2_inputs_sorted, m2_outputs, m2_outputs_sorted)

    # reunite the matrices
    result = twisted_multiplication(m1_ordered, m2_ordered, covering)

    # prepare lists to reorder the result
    m2_inputs_from_inputs_sorted = sorted(m2_inputs_from_inputs, key=lambda input_m2: input_m2.index)
    group_inputs = m1_inputs_sorted + m2_inputs_from_inputs_sorted

    m1_outputs_to_output_sorted = sorted(m1_outputs_to_output, key=lambda output_m1: output_m1.index)
    group_outputs = m1_outputs_to_output_sorted + m2_outputs_sorted

    # order result
    result_ordered = order_matrix(result, inputs, group_inputs, outputs, group_outputs)

    return result_ordered


def remove_cup(m: GenericMatrix, index1: int, index2: int) -> GenericMatrix:
    """Transforms *m* to remove the cup from *index1* to *index2*.

    Args:
        m (GenericMatrix): matrix to transform
        index1 (int): first index of the cup
        index2 (int): second index of the cup

    Returns:
        GenericMatrix: *m* with its number of lines divided by 4 and added as should be (see conception document for
            more information)
    """
    height = m.shape[0]
    base_size = height.bit_length()
    if index1 not in np.arange(base_size):
        raise ValueError("Index out of bound, m output base size : %d and index1 given : %d" % (base_size, index1))
    if index2 not in np.arange(base_size):
        raise ValueError("Index out of bound, m output base size : %d and index2 given : %d" % (base_size, index2))
    list_0 = []  # contains the lines for which the bit number index1 and number index2 are both equal to 0
    list_1 = []  # contains the lines for which the bit number index1 and number index2 are both equal to 1
    for _i in np.arange(height):
        _i = EnhancedInt(_i)  # i is given in little endian, but it's easier to think in big endian for the wires
        if _i[base_size - 1 - index1] == _i[base_size - 1 - index2] == 0:
            list_0.append(_i)
        if _i[base_size - 1 - index1] == _i[base_size - 1 - index2] == 1:
            list_1.append(_i)
    return m[list_0, :] + m[list_1, :]  # returns the lines described above split in two matrices and added


def remove_cap(m: GenericMatrix, index1: int, index2: int) -> GenericMatrix:
    """Transforms *m* to remove the cap from *index1* to *index2*.

    Args:
        m (GenericMatrix): matrix to transform
        index1 (int): first index of the cap
        index2 (int): second index of the cap

    Returns:
        GenericMatrix: *m* with its number of columns divided by 4 and added as should be (see conception document for
            more information)
    """
    width = m.shape[0]
    base_size = width.bit_length()
    if index1 not in np.arange(base_size):
        raise ValueError("Index out of bound, m output base size : %d and index1 given : %d" % (base_size, index1))
    if index2 not in np.arange(base_size):
        raise ValueError("Index out of bound, m output base size : %d and index2 given : %d" % (base_size, index2))
    list_0 = []  # contains the lines for which the bit number index1 and number index2 are both equal to 0
    list_1 = []  # contains the lines for which the bit number index1 and number index2 are both equal to 1
    for _i in np.arange(width):
        _i = EnhancedInt(_i)  # i is given in little endian, but it's easier to think in big endian for the wires
        if _i[base_size - 1 - index1] == _i[base_size - 1 - index2] == 0:
            list_0.append(_i)
        if _i[base_size - 1 - index1] == _i[base_size - 1 - index2] == 1:
            list_1.append(_i)
    return m[:, list_0] + m[:, list_1]  # returns the lines described above split in two matrices and added


def input_to_output(m: GenericMatrix, input_index: int, output_index: int = -1) -> GenericMatrix:
    """Transforms *m* to have the input connected to *input_index* becoming an output connected to to *output_index*.

    Args:
        m (GenericMatrix): matrix to transform
        input_index (int): index of the output being taken off
        output_index (int): index of the output being added

    Returns:
        GenericMatrix: *m* with its number of columns divided by 2, its number of lines multiplied by 2 and the
            coefficients moved accordingly (see conception document for more information)
    """
    _height, _width = m.shape
    _input_base_size = _width.bit_length() - 1
    _new_input_base_size = _input_base_size - 1
    _output_base_size = _height.bit_length() - 1
    _new_output_base_size = _output_base_size + 1
    if input_index not in np.arange(_input_base_size):
        raise ValueError("Index out of bound, m input base size : %d and index1 given : %d" % (_input_base_size,
                                                                                               input_index))
    if output_index not in list(np.arange(_new_output_base_size)) + list(np.arange(-_new_output_base_size, 0)):
        raise ValueError("Index out of bound, m output base size : %d and index2 given : %d" % (_output_base_size,
                                                                                                output_index))
    try:
        _result = UsedFragment(np.zeros((2 ** _new_output_base_size, 2 ** _new_input_base_size)), z=m.z)
    except AttributeError:
        _result = UsedFragment(np.zeros((2 ** _new_output_base_size, 2 ** _new_input_base_size)))

    def _transformation(_i: EnhancedInt, _j: EnhancedInt, in_index: int, out_index: int):
        """
        Args:
            _i: represent the output to transform
            _j: represent the input to transform
            in_index: index of the input to be taken
            out_index: index of the destination

        Returns:

        """
        _i.insert(out_index, _j[in_index])
        del _j[in_index]
        return _i, _j

    _matrix = UsedFragment(m)
    for _i in np.arange(2 ** _output_base_size):
        for _j in np.arange(2 ** _input_base_size):
            _enhanced_i = EnhancedInt(_i)
            _enhanced_j = EnhancedInt(_j)
            if output_index < 0:
                _output_index = _new_output_base_size + output_index
            else:
                _output_index = output_index
            # the following operations with the _input_base_size and the _new_output_base_size are needed because
            # _transformation treats the numbers in little endian and the wires order is given in big endian
            _position = _transformation(_enhanced_i, _enhanced_j, _input_base_size - 1 - input_index,
                                        _new_output_base_size - 1 - _output_index)
            _result[_position] = _matrix[_i, _j]
    return _result


def output_to_input(m: GenericMatrix, output_index: int, input_index: int = -1) -> GenericMatrix:
    """Transforms *m* to have the output connected to *output_index* becoming an input connected to to *input_index*.

    Args:
        m (GenericMatrix): matrix to transform
        output_index (int): index of the output being taken off
        input_index (int): index of the output being added

    Returns:
        GenericMatrix: *m* with its number of lines divided by 2, its number of columns multiplied by 2 and the
            coefficients moved accordingly (see conception document for more information)
    """
    height, width = m.shape
    input_base_size = width.bit_length() - 1
    new_input_base_size = input_base_size + 1
    output_base_size = height.bit_length() - 1
    new_output_base_size = output_base_size - 1
    if input_index not in list(np.arange(new_input_base_size)) + list(np.arange(-new_input_base_size, 0)):
        raise ValueError("Index out of bound, m input base size : %d and index1 given : %d" % (input_base_size,
                                                                                               input_index))
    if output_index not in np.arange(output_base_size):
        raise ValueError("Index out of bound, m output base size : %d and index2 given : %d" % (output_base_size,
                                                                                                output_index))
    try:
        _result = UsedFragment(np.zeros((2 ** new_output_base_size, 2 ** new_input_base_size)), z=m.z)
    except AttributeError:
        _result = UsedFragment(np.zeros((2 ** new_output_base_size, 2 ** new_input_base_size)))

    def _transformation(_i: EnhancedInt, _j: EnhancedInt, in_index: int, out_index: int):
        """

        Args:
            _i: represent the output to transform
            _j: represent the input to transform
            in_index: index of the destination
            out_index: index of the output to be taken

        Returns:

        """

        _j.insert(in_index, _i[out_index])
        del _i[out_index]
        return _i, _j

    _matrix = UsedFragment(m)
    for _i in np.arange(2 ** output_base_size):
        for _j in np.arange(2 ** input_base_size):
            _enhanced_i = EnhancedInt(_i)
            _enhanced_j = EnhancedInt(_j)
            if input_index < 0:
                _input_index = new_input_base_size + input_index
            else:
                _input_index = input_index
            # the following operations with the input_base_size and the new_output_base_size are needed because
            # _transformation treats the numbers in little endian and the wires order is given in big endian
            _position = _transformation(_enhanced_i, _enhanced_j, new_input_base_size - 1 - _input_index,
                                        output_base_size - 1 - output_index)
            _result[_position] = _matrix[_i, _j]
    return _result


def order_matrix(m: GenericMatrix, input_order: list, matrix_input_order: list,
                 output_order: list, matrix_output_order: list) -> GenericMatrix:
    """Transforms the matrix so the order of the I/O is change accordingly to the args. This is only a permutation on
    the lines and the columns.

    Args:
        m (GenericMatrix): initial matrix
        input_order (list): order of the inputs as they are expected
        matrix_input_order (list): order of the inputs on the initial matrix
        output_order (list): order of the outputs as they are expected
        matrix_output_order (list): order of the outputs on the initial matrix

    Returns:
        GenericMatrix: matrix of the same size but with.
    """
    if len(input_order) != len(matrix_input_order):
        raise ValueError('input_order and matrix_input_order length differ')
    if len(output_order) != len(matrix_output_order):
        raise ValueError('output_order and matrix_output_order length differ')
    _height, _width = m.shape
    _input_base_size = max(_width.bit_length() - 1, 0)
    _output_base_size = max(_height.bit_length() - 1, 0)
    _result = deepcopy(UsedFragment(m))

    def _permutation_dictionary(pre_permutation_list, post_permutation_list):
        length = len(pre_permutation_list)
        permutation_dic = {}
        for i in np.arange(length):
            for j in np.arange(length):
                # as before, since we are in little endian representation, we need to switch the order for the wires
                if pre_permutation_list[length - 1 - i] == post_permutation_list[length - 1 - j]:
                    permutation_dic[i] = j
                    break
        return permutation_dic

    _input_dict = _permutation_dictionary(input_order, matrix_input_order)
    _output_dict = _permutation_dictionary(output_order, matrix_output_order)

    def _transformation(_i: EnhancedInt, _permutation: dict):
        length = len(_permutation)
        image = EnhancedInt(0)
        for _k in np.arange(length):
            image[_k] = _i[_permutation[_k]]
        return image

    _matrix = deepcopy(UsedFragment(m))
    for _i in np.arange(2 ** _output_base_size):
        _enhanced_i = EnhancedInt(_i)
        _new_i = _transformation(_enhanced_i, _output_dict)
        for _j in np.arange(2 ** _input_base_size):
            _enhanced_j = EnhancedInt(_j)
            # the following operations with the _input_base_size and the _new_output_base_size are needed because
            # _transformation treats the numbers in little endian and the wires order is given in big endian
            _new_j = _transformation(_enhanced_j, _input_dict)
            _result[_new_i, _new_j] = _matrix[_i, _j]

    return _result


def twisted_multiplication(m1: GenericMatrix, m2: GenericMatrix, covering: int) -> GenericMatrix:
    """Core of this module, this section is half way between the serialization of the two matrices (one being put after
    the other) and the parallelisation of the matrices (one being put next to the other). This means that you can have
    wires going from a matrix to another, wires directly inputting to the second matrix and wire directly outputting
    from the first matrix (see conception document for more information)

    Args:
        m1 (GenericMatrix): first matrix
        m2 (GenericMatrix): second matrix
        covering (int): number of wires linking the two matrices

    Returns:
        GenericMatrix: matrix resulting of this operation
    """
    list_m1 = []  # type: List[UsedFragment]
    list_m2 = []  # type: List[UsedFragment]
    height_m1, width_m1 = EnhancedInt(m1.shape[0]), EnhancedInt(m1.shape[1])
    height_m2, width_m2 = EnhancedInt(m2.shape[0]), EnhancedInt(m2.shape[1])
    if height_m1.count(1) != 1 or width_m1.count(1) != 1:
        raise ValueError('Matrices shape should be perfect powers of 2, but (%d,%d) received for m1' % (height_m1,
                                                                                                        width_m1))
    if height_m2.count(1) != 1 or width_m2.count(1) != 1:
        raise ValueError('Matrices shape should be perfect powers of 2, but (%d,%d) received for m2' % (height_m1,
                                                                                                        width_m1))
    for i in np.arange(height_m1 >> covering):
        list_m1.append(m1[i * 2 ** covering:(i + 1) * 2 ** covering, :])
    for i in np.arange(width_m2 >> covering):
        list_m2.append(m2[:, i::width_m2 >> covering])

    list_m12 = []  # type: List[List[UsedFragment]]
    for i in np.arange(height_m1 >> covering):
        list_m12_i = []  # type: List[UsedFragment]
        for j in np.arange(width_m2 >> covering):
            list_m12_i.append(list_m2[j].dot(list_m1[i]))
        list_m12.append(list_m12_i)

    result = UsedFragment(np.zeros(((height_m1 * height_m2) >> covering, (width_m2 * width_m1) >> covering)))
    for i in np.arange(height_m1 >> covering):
        for j in np.arange(width_m2 >> covering):
            result[i * height_m2: (i + 1) * height_m2, j::width_m2 >> covering] = list_m12[i][j]
    return result
