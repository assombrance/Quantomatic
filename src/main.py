import sys

import numpy as np

import loader

from data import Pi4Matrix, Node

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)

inputList, outputList = loader.interpret_i_o(sys.argv)

loader.check_for_doubles(inputList, outputList)

assumed_order_e, inputList, outputList, matrix = loader.main(sys.argv[1], inputList, outputList)

if assumed_order_e:
    print('Caution, not all wires given explicitly as input or output, assuming following order:')
    print('Inputs: ' + str(inputList))
    print('Outputs: ' + str(outputList))
print('_______________')
matrix.normalize()
print(matrix)
