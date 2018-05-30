# coding=UTF-8
"""
Script called by quantomatic, it itself calls the :ref:`loader` to load and execute the computation functions.

Args:
    graph_path, inputs, outputs:
        *graph_path*: The path to the target graph. Can be a relative path or an absolute path.

        *inputs*: List of input names (wire names more precisely). Separated by commas, surrounded by brackets

        *outputs*: List of output names (wire names more precisely). Separated by commas, surrounded by brackets
Returns:
    the computed matrix, if the I/O order was not given explicitly, the matrix is preceded by a warning message
    and the order taken to compute the matrix, this message is terminated by *'_______________'*
"""
import sys

import numpy as np

import loader


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=sys.maxsize)

    inputList, outputList = loader.interpret_i_o(sys.argv)

    loader.check_for_doubles(inputList, outputList)

    # tmp
    for _ in np.arange(5):
        assumed_order_e, inputList, outputList, matrix = loader.main(sys.argv[1], inputList, outputList)

        if assumed_order_e:
            print('Caution, not all wires given explicitly as input or output, assuming following order:')
            print('Inputs: ' + str(inputList))
            print('Outputs: ' + str(outputList))
        print('_______________')
        matrix.normalize()
        print(matrix)
