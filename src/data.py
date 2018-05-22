# from __future__ import annotations
import inspect
import math
from copy import deepcopy
from typing import List, TypeVar, Union, SupportsInt

import numpy as np


class PrettyStr:
    def __str__(self) -> str:
        string = "<" + self.__class__.__name__ + ": "
        for atr in sorted(self.__dict__):
            string += str(self.__getattribute__(atr)) + " (" + atr + "), "
        string = string[:-2] + ">"
        return string


class AtrComparison:
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class Wire(PrettyStr, AtrComparison):
    def __init__(self, name: str) -> None:
        self.name = name


class Node(Wire):
    def __init__(self, name: str, angle: float = 0., node_type: str = 'Z', arity: int = 0) -> None:
        super().__init__(name)
        self.angle = angle
        self.arity = arity
        self.node_type = node_type


class Edge(PrettyStr, AtrComparison):
    def __init__(self, name: str, n1: Wire, n2: Wire, label: str = "") -> None:
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.label = label

    def __iter__(self):
        yield self.n1
        yield self.n2


class ConnectionPoint(PrettyStr, AtrComparison):
    def __init__(self, is_matrix_2=False, is_out=False, index=0) -> None:
        self.is_matrix_2 = is_matrix_2
        self.is_out = is_out
        self.index = index


class InterMatrixLink(PrettyStr, AtrComparison):
    def __init__(self, point1: ConnectionPoint, point2: ConnectionPoint) -> None:
        self.point1 = point1
        self.point2 = point2


class AbstractMatrix(AtrComparison, np.matrix):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        son = kwargs.get("son")
        if not son:
            raise NotImplementedError("Class %s doesn't implement %s()" %
                                      (self.__class__.__name__, inspect.stack()[0][3]))

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (cls, inspect.stack()[0][3]))

    def __array_finalize__(self, obj):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def dot(self, b, out=None):
        if isinstance(self, b.__class__):
            raise NotImplementedError("Class %s doesn't implement %s()" %
                                      (self.__class__.__name__, inspect.stack()[0][3]))
        return NotImplemented

    def __getitem__(self, key):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __setitem__(self, key, value):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __delitem__(self, key):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    @classmethod
    def node_to_matrix(cls, node: Node, in_number: int, out_number: int):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (cls.__name__, inspect.stack()[0][3]))

    @property
    def shape(self):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    @property
    def size(self):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def tensor_product(self, b):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def tensor_power(self, power: int):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __copy__(self, *args):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __deepcopy__(self, *args):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __add__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __iadd__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __radd__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __sub__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __isub__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __rsub__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __pow__(self, n: int):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __mul__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __imul__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __rmul__(self, y):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def any(self, axis=None, out=None):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def all(self, axis=None, out=None):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def reshape(self, shape, *shapes, order='C'):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __neg__(self, *args, **kwargs):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __pos__(self, *args, **kwargs):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __eq__(self, other):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __str__(self, *args, **kwargs):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def __repr__(self, *args, **kwargs):
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))


class Pi4Matrix(AbstractMatrix):
    def __new__(cls, m0, m1=None, m2=None, m3=None, z: int = 0):
        if isinstance(m0, Pi4Matrix):
            return m0
        obj = np.asarray([0], dtype=int).view(cls)
        obj.m0 = np.asarray(m0, dtype=int)
        if m1 is None:
            obj.m1 = np.zeros_like(m0, dtype=int)
        else:
            obj.m1 = np.asarray(m1, dtype=int)
        if m2 is None:
            obj.m2 = np.zeros_like(m0, dtype=int)
        else:
            obj.m2 = np.asarray(m2, dtype=int)
        if m3 is None:
            obj.m3 = np.zeros_like(m0, dtype=int)
        else:
            obj.m3 = np.asarray(m3, dtype=int)
        obj.z = z
        return obj

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, son=True, **kwargs)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.m0 = getattr(obj, 'm0', None)
        self.m1 = getattr(obj, 'm1', None)
        self.m2 = getattr(obj, 'm2', None)
        self.m3 = getattr(obj, 'm3', None)
        self.z = getattr(obj, 'z', None)

    @property
    def shape(self):
        return self.m0.shape

    @property
    def size(self):
        return self.m0.size

    @classmethod
    def node_to_matrix(cls, node: Node, in_number: int, out_number: int):
        # classic_hadamard = 1/2( [[1, 1], [1, -1]]*exp(i*pi/4) + [[-1, -1], [-1, 1]]*exp(3i*pi/4) )
        # comes from the fact that exp(i*pi/4)-exp(3i*pi/4)=sqrt(2)
        fragment = 4
        if in_number < 0 or out_number < 0:
            raise ValueError('in_number and out_number must be positive integers')
        if node.node_type == 'hadamard':
            if in_number + out_number != 2:
                raise ValueError('Hadamard gate is only known for node with an arity of 2')
            base = np.ones((1 << out_number, 1 << in_number))
            base[-1, -1] = -1
            null_base = np.zeros((1 << out_number, 1 << in_number))
            result = Pi4Matrix(null_base, base, null_base, base, 1)
        elif node.node_type == 'Z' or node.node_type == 'X':
            position = node.angle * fragment
            if position in np.arange(2*fragment):
                null_base = np.zeros((1 << out_number, 1 << in_number))
                base = deepcopy(null_base)
                base[0, 0] = 1
                matrices = [base]
                for _ in np.arange(fragment):
                    matrices.append(deepcopy(null_base))
                if position < fragment:
                    matrices[int(position)][-1, -1] = 1
                else:
                    matrices[int(position)][-1, -1] = -1
                result = Pi4Matrix(matrices[0], matrices[1], matrices[2], matrices[3], 0)
            else:
                raise ValueError('You are trying to create a Pi4Matrix from a node with an angle which is not a '
                                 'multiple of pi/4 (angle: ' + str(node.angle) + ')')
            if node.node_type == 'X':
                base = np.ones((2, 2))
                base[-1, -1] = -1
                null_base = np.zeros((2, 2))
                h = Pi4Matrix(null_base, base, null_base, -base, 1)
                result = (h.tensor_power(out_number)).dot(result).dot(h.tensor_power(in_number))
        else:
            raise ValueError('Unknown node type: %s' % node.node_type)
        return result

    def normalize(self):
        smallest_2_power = 32
        non_zero_coefficient_found = False
        height, width = self.shape
        for i in np.arange(height):
            for j in np.arange(width):
                x = EnhancedInt(abs(self.m0[i, j]))
                if x.index(1) > 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
                x = EnhancedInt(abs(self.m1[i, j]))
                if x.index(1) > 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
                x = EnhancedInt(abs(self.m2[i, j]))
                if x.index(1) > 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
                x = EnhancedInt(abs(self.m3[i, j]))
                if x.index(1) > 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
        if not non_zero_coefficient_found:
            # noinspection PyAttributeOutsideInit
            self.z = 0
        if smallest_2_power > 0:
            self.z -= smallest_2_power
            for i in np.arange(height):
                for j in np.arange(width):
                    self.m0[i, j] >>= smallest_2_power
                    self.m1[i, j] >>= smallest_2_power
                    self.m2[i, j] >>= smallest_2_power
                    self.m3[i, j] >>= smallest_2_power

    def tensor_product(self, b):
        """Computes the tensor product of matrix *a* and *b*
        *a* and *b* have to be matrices (numpy matrix or 2-D array)
        Args:
            b (GenericMatrix): Second argument
        Returns:
            GenericMatrix: Tensor product of a and b
        """
        ma, na = self.shape
        mb, nb = b.shape
        mr, nr = ma * mb, na * nb
        result = Pi4Matrix(np.zeros((mr, nr)))
        for i in np.arange(mr):
            for j in np.arange(nr):
                result[i, j] = self[i // mb, j // nb] * b[i % mb, j % nb]
        return result

    def tensor_power(self, power: int):
        """Computes the *a**power*, in the tensor sense
        *a* has to be a matrix (numpy matrix or 2-D array)
        *power* must be positive (or equal to 0)
        Args:
            power (int): The power to elevate a to
        Returns:
            GenericMatrix: 'power'
        """
        if power < 0:
            raise NameError('Tensor power not defined for a negative power')
        if power == 0:
            return Pi4Matrix(np.identity(1))
        else:
            return self.tensor_product(self.tensor_power(power - 1))

    def __copy__(self, *args):
        new_one = type(self)(0)
        new_one.__dict__.update(self.__dict__)
        return new_one

    def __deepcopy__(self, *args):
        cls = self.__class__  # type: Pi4Matrix
        result = cls.__new__(cls, 0)
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v))
        return result

    def dot(self, b, out=None) -> 'Pi4Matrix':
        _pi_4_b = Pi4Matrix(b)
        if _pi_4_b.shape[0] == self.shape[1]:
            _z = _pi_4_b.z + self.z
            _m0 = self.m0.dot(_pi_4_b.m0) - self.m1.dot(_pi_4_b.m3) - self.m2.dot(_pi_4_b.m2) - self.m3.dot(_pi_4_b.m1)
            _m1 = self.m0.dot(_pi_4_b.m1) + self.m1.dot(_pi_4_b.m0) - self.m2.dot(_pi_4_b.m3) - self.m3.dot(_pi_4_b.m2)
            _m2 = self.m0.dot(_pi_4_b.m2) + self.m1.dot(_pi_4_b.m1) + self.m2.dot(_pi_4_b.m0) - self.m3.dot(_pi_4_b.m3)
            _m3 = self.m0.dot(_pi_4_b.m3) + self.m1.dot(_pi_4_b.m2) + self.m2.dot(_pi_4_b.m1) + self.m3.dot(_pi_4_b.m0)
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        raise ValueError('shapes (%d,%d) and (%d,%d) not aligned: %d (dim 1) != %d (dim 0)' % (self.shape[0],
                                                                                               self.shape[1],
                                                                                               _pi_4_b.shape[0],
                                                                                               _pi_4_b.shape[1],
                                                                                               self.shape[1],
                                                                                               _pi_4_b.shape[0]))

    def __add__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            if self.z < _pi_4_y.z:
                _z = self.z
                _m0 = self.m0 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m0
                _m1 = self.m1 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m1
                _m2 = self.m2 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m2
                _m3 = self.m3 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m3
            else:
                _z = _pi_4_y.z
                _m0 = self.m0 + _pi_4_y.m0 * (2 ** (self.z - _pi_4_y.z))
                _m1 = self.m1 + _pi_4_y.m1 * (2 ** (self.z - _pi_4_y.z))
                _m2 = self.m2 + _pi_4_y.m2 * (2 ** (self.z - _pi_4_y.z))
                _m3 = self.m3 + _pi_4_y.m3 * (2 ** (self.z - _pi_4_y.z))
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        elif type(y) == int:
            _m0 = self.m0 + y
            return Pi4Matrix(_m0, self.m1, self.m2, self.m3, self.z)
        elif _pi_4_y.shape == ():
            raise ValueError('+ is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __iadd__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            if self.z < _pi_4_y.z:
                self.m0 = self.m0 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m0
                self.m1 = self.m1 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m1
                self.m2 = self.m2 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m2
                self.m3 = self.m3 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m3
            else:
                self.m0 = self.m0 + _pi_4_y.m0 * (2 ** (self.z - _pi_4_y.z))
                self.m1 = self.m1 + _pi_4_y.m1 * (2 ** (self.z - _pi_4_y.z))
                self.m2 = self.m2 + _pi_4_y.m2 * (2 ** (self.z - _pi_4_y.z))
                self.m3 = self.m3 + _pi_4_y.m3 * (2 ** (self.z - _pi_4_y.z))
                self.z = _pi_4_y.z
            return self
        elif type(y) == int:
            self.m0 += y
            return self
        elif _pi_4_y.shape == ():
            raise ValueError('+ is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __radd__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            if self.z < _pi_4_y.z:
                _z = self.z
                _m0 = self.m0 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m0
                _m1 = self.m1 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m1
                _m2 = self.m2 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m2
                _m3 = self.m3 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m3
            else:
                _z = _pi_4_y.z
                _m0 = self.m0 + _pi_4_y.m0 * (2 ** (self.z - _pi_4_y.z))
                _m1 = self.m1 + _pi_4_y.m1 * (2 ** (self.z - _pi_4_y.z))
                _m2 = self.m2 + _pi_4_y.m2 * (2 ** (self.z - _pi_4_y.z))
                _m3 = self.m3 + _pi_4_y.m3 * (2 ** (self.z - _pi_4_y.z))
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        elif type(y) == int:
            _m0 = self.m0 + y
            return Pi4Matrix(_m0, self.m1, self.m2, self.m3, self.z)
        elif _pi_4_y.shape == ():
            raise ValueError('+ is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __sub__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            if self.z < _pi_4_y.z:
                _z = self.z
                _m0 = self.m0 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m0
                _m1 = self.m1 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m1
                _m2 = self.m2 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m2
                _m3 = self.m3 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m3
            else:
                _z = _pi_4_y.z
                _m0 = self.m0 - _pi_4_y.m0 * (2 ** (self.z - _pi_4_y.z))
                _m1 = self.m1 - _pi_4_y.m1 * (2 ** (self.z - _pi_4_y.z))
                _m2 = self.m2 - _pi_4_y.m2 * (2 ** (self.z - _pi_4_y.z))
                _m3 = self.m3 - _pi_4_y.m3 * (2 ** (self.z - _pi_4_y.z))
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        elif type(y) == int:
            _m0 = self.m0 - y
            return Pi4Matrix(_m0, self.m1, self.m2, self.m3, self.z)
        elif _pi_4_y.shape == ():
            raise ValueError('- is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __isub__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            if self.z < _pi_4_y.z:
                self.m0 = self.m0 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m0
                self.m1 = self.m1 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m1
                self.m2 = self.m2 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m2
                self.m3 = self.m3 * (2 ** (_pi_4_y.z - self.z)) - _pi_4_y.m3
            else:
                self.m0 = self.m0 - _pi_4_y.m0 * (2 ** (self.z - _pi_4_y.z))
                self.m1 = self.m1 - _pi_4_y.m1 * (2 ** (self.z - _pi_4_y.z))
                self.m2 = self.m2 - _pi_4_y.m2 * (2 ** (self.z - _pi_4_y.z))
                self.m3 = self.m3 - _pi_4_y.m3 * (2 ** (self.z - _pi_4_y.z))
                self.z = _pi_4_y.z
            return self
        elif type(y) == int:
            self.m0 -= y
            return self
        elif _pi_4_y.shape == ():
            raise ValueError('- is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __rsub__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            if self.z < _pi_4_y.z:
                _z = self.z
                _m0 = - self.m0 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m0
                _m1 = - self.m1 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m1
                _m2 = - self.m2 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m2
                _m3 = - self.m3 * (2 ** (_pi_4_y.z - self.z)) + _pi_4_y.m3
            else:
                _z = _pi_4_y.z
                _m0 = - self.m0 + _pi_4_y.m0 * (2 ** (self.z - _pi_4_y.z))
                _m1 = - self.m1 + _pi_4_y.m1 * (2 ** (self.z - _pi_4_y.z))
                _m2 = - self.m2 + _pi_4_y.m2 * (2 ** (self.z - _pi_4_y.z))
                _m3 = - self.m3 + _pi_4_y.m3 * (2 ** (self.z - _pi_4_y.z))
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        elif type(y) == int:
            _m0 = - self.m0 + y
            return Pi4Matrix(_m0, self.m1, self.m2, self.m3, self.z)
        elif _pi_4_y.shape == ():
            raise ValueError('- is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __mul__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            _z = self.z + _pi_4_y.z
            _m0 = self.m0 * _pi_4_y.m0 - self.m1 * _pi_4_y.m3 - self.m2 * _pi_4_y.m2 - self.m3 * _pi_4_y.m1
            _m1 = self.m0 * _pi_4_y.m1 + self.m1 * _pi_4_y.m0 - self.m2 * _pi_4_y.m3 - self.m3 * _pi_4_y.m2
            _m2 = self.m0 * _pi_4_y.m2 + self.m1 * _pi_4_y.m1 + self.m2 * _pi_4_y.m0 - self.m3 * _pi_4_y.m3
            _m3 = self.m0 * _pi_4_y.m3 + self.m1 * _pi_4_y.m2 + self.m2 * _pi_4_y.m1 + self.m3 * _pi_4_y.m0
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        elif type(y) == int:
            _m0 = self.m0 * y
            _m1 = self.m1 * y
            _m2 = self.m2 * y
            _m3 = self.m3 * y
            return Pi4Matrix(_m0, _m1, _m2, _m3, self.z)
        elif _pi_4_y.shape == ():
            raise ValueError('* is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __imul__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            self.z += _pi_4_y.z
            self.m0 = self.m0 * _pi_4_y.m0 - self.m1 * _pi_4_y.m3 - self.m2 * _pi_4_y.m2 - self.m3 * _pi_4_y.m1
            self.m1 = self.m0 * _pi_4_y.m1 + self.m1 * _pi_4_y.m0 - self.m2 * _pi_4_y.m3 - self.m3 * _pi_4_y.m2
            self.m2 = self.m0 * _pi_4_y.m2 + self.m1 * _pi_4_y.m1 + self.m2 * _pi_4_y.m0 - self.m3 * _pi_4_y.m3
            self.m3 = self.m0 * _pi_4_y.m3 + self.m1 * _pi_4_y.m2 + self.m2 * _pi_4_y.m1 + self.m3 * _pi_4_y.m0
            return self
        elif type(y) == int:
            self.m0 *= y
            self.m1 *= y
            self.m2 *= y
            self.m3 *= y
            return self
        elif _pi_4_y.shape == ():
            raise ValueError('* is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __rmul__(self, y):
        _pi_4_y = Pi4Matrix(y)
        if _pi_4_y.shape == self.shape:
            _z = self.z + _pi_4_y.z
            _m0 = _pi_4_y.m0 * self.m0 - _pi_4_y.m3 * self.m1 - _pi_4_y.m2 * self.m2 - _pi_4_y.m1 * self.m3
            _m1 = _pi_4_y.m1 * self.m0 + _pi_4_y.m0 * self.m1 - _pi_4_y.m3 * self.m2 - _pi_4_y.m2 * self.m3
            _m2 = _pi_4_y.m2 * self.m0 + _pi_4_y.m1 * self.m1 + _pi_4_y.m0 * self.m2 - _pi_4_y.m3 * self.m3
            _m3 = _pi_4_y.m3 * self.m0 + _pi_4_y.m2 * self.m1 + _pi_4_y.m1 * self.m2 + _pi_4_y.m0 * self.m3
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        elif type(y) == int:
            _m0 = y * self.m0
            _m1 = y * self.m1
            _m2 = y * self.m2
            _m3 = y * self.m3
            return Pi4Matrix(_m0, _m1, _m2, _m3, self.z)
        elif _pi_4_y.shape == ():
            raise ValueError('* is only defined for matrix and int')
        else:
            raise ValueError('Size mismatch')

    def __pow__(self, n: int):
        _result = Pi4Matrix(self)
        for _ in np.arange(n):
            _result = _result.dot(self)
        return _result

    def __rpow__(self, n: int):
        _result = Pi4Matrix(self)
        for _ in np.arange(n):
            _result = _result.dot(self)
        self.m0 = _result.m0
        self.m1 = _result.m1
        self.m2 = _result.m2
        self.m3 = _result.m3
        self.z = _result.z
        return self

    def any(self, axis=None, out=None):
        return self.m0.any(axis, out) or self.m1.any(axis, out) or self.m2.any(axis, out) or self.m3.any(axis, out)

    def all(self, axis=None, out=None):
        return self.m0.any(axis, out) and self.m1.any(axis, out) and self.m2.any(axis, out) and self.m3.any(axis, out)

    def reshape(self, shape, *shapes, order='C'):
        _m0 = self.m0.reshape(shape, *shapes, order)
        _m1 = self.m1.reshape(shape, *shapes, order)
        _m2 = self.m2.reshape(shape, *shapes, order)
        _m3 = self.m3.reshape(shape, *shapes, order)
        return Pi4Matrix(_m0, _m1, _m2, _m3, self.z)

    def __neg__(self, *args, **kwargs):
        return Pi4Matrix(-self.m0, -self.m1, -self.m2, -self.m3, self.z)

    def __pos__(self, *args, **kwargs):
        return Pi4Matrix(self)

    def __eq__(self, other):
        return not (self - other).any()

    def __getitem__(self, key):
        return Pi4Matrix(self.m0[key], self.m1[key], self.m2[key], self.m3[key], self.z)

    def __setitem__(self, key, value):
        """Set *self[key]* to *value*

        Args:
            key:
            value (Pi4Matrix):

        Returns:
            None
        """
        if not isinstance(value, Pi4Matrix):
            return ValueError("Value should be a matrix")
        if self.z < value.z:
            self.z = value.z
            self.m0 = self.m0 * (2 ** (value.z - self.z))
            self.m0[key] = value.m0
            self.m1 = self.m1 * (2 ** (value.z - self.z))
            self.m1[key] = value.m1
            self.m2 = self.m2 * (2 ** (value.z - self.z))
            self.m2[key] = value.m2
            self.m3 = self.m3 * (2 ** (value.z - self.z))
            self.m3[key] = value.m3
        else:
            _key = eval(repr(key))  # it looks stupid, and it is : a super weird bug causes the key to be poorly read,
            # but only in the __setitem__ for self.m0 (and others), this is the only fix I found, but I'll try to fix it
            self.m0[_key] = value.m0 * (2 ** (self.z - value.z))
            self.m1[_key] = value.m1 * (2 ** (self.z - value.z))
            self.m2[_key] = value.m2 * (2 ** (self.z - value.z))
            self.m3[_key] = value.m3 * (2 ** (self.z - value.z))

    def __delitem__(self, key):
        del self.m0[key]
        del self.m1[key]
        del self.m2[key]
        del self.m3[key]

    def __str__(self, *args, **kwargs):
        _m0_lines = str(self.m0).replace('\n', ' \n').split('\n')
        _m1_lines = str(self.m1).replace('\n', ' \n').split('\n')
        _m2_lines = str(self.m2).replace('\n', ' \n').split('\n')
        _m3_lines = str(self.m3).replace('\n', ' \n').split('\n')
        string = ""
        half_factor = "(1/2^" + str(self.z) + ")*("
        padding_needed = len(half_factor)
        padding = " " * padding_needed
        for _i in np.arange(self.m0.shape[0] - 1):
            string += padding + _m0_lines[_i] + "   " + _m1_lines[_i] + "               " + _m2_lines[_i] + \
                      "                " + _m3_lines[_i] + "\n"
        string += "(1/2^" + str(self.z) + ")*(" + _m0_lines[-1] + " + " + _m1_lines[-1] + "*exp(i*pi/4) + " + \
                  _m2_lines[-1] + "*exp(2i*pi/4) + " + _m3_lines[-1] + "*exp(3i*pi/4) )\n"
        return string

    def __repr__(self, *args, **kwargs):
        string = "(1/(2**" + str(self.z) + "))*(np." + repr(self.m0) + " + np." + repr(self.m1) + \
                 "*np.exp(1j*math.pi/4) + np." + repr(self.m2) + "*np.exp(1j*math.pi/2) + np." + \
                 repr(self.m3) + "*np.exp(3j*math.pi/4))"
        string = string.replace('\n', '').replace(' ', '')
        return string


class EnhancedInt(int):
    def __init__(self, x: Union[str, bytes, SupportsInt] = ...) -> None:
        int.__init__(int(x))
        self.__value__ = int(x)

    @staticmethod
    def from_list(x: List[int]):
        _value = 0
        if isinstance(x, List):
            for i in x:
                if i not in [0, 1]:
                    raise ValueError('List must be composed of binary numbers (ie 0 or 1)')
            for _index in np.arange(len(x)):
                _value += x[_index] << _index
        else:
            _value = x
        return EnhancedInt(_value)

    def __add__(self, other) -> int:
        return self.__value__ + other

    def __divmod__(self, other):
        return divmod(self.__value__, other)

    def __mul__(self, other):
        return self.__value__ * other

    def __hash__(self):
        return hash(self.__value__)

    def __abs__(self):
        return abs(self.__value__)

    def __bool__(self):
        return bool(self.__value__)

    def __bytes__(self):
        return bytes(self.__value__)

    def __complex__(self):
        return complex(self.__value__)

    def __float__(self):
        return float(self.__value__)

    def __floor__(self):
        return math.floor(self.__value__)

    def __ge__(self, other):
        return self.__value__ >= other

    def __le__(self, other):
        return self.__value__ <= other

    def __eq__(self, other):
        return self.__value__ == other

    def __lt__(self, other):
        return self.__value__ < other

    def __gt__(self, other):
        return self.__value__ > other

    def __ne__(self, other):
        return self.__value__ != other

    def __format__(self, format_spec):
        return format(self.__value__, format_spec)

    def __get__(self, instance, owner):
        return self.__value__

    def __hex__(self):
        return hex(self.__value__)

    def __iadd__(self, other):
        self.__value__ += other
        return self

    def __iand__(self, other):
        self.__value__ &= other
        return self

    def __idiv__(self, other):
        self.__value__ /= other
        return self

    def __imod__(self, other):
        self.__value__ %= other
        return self

    def __ifloordiv__(self, other):
        self.__value__ //= other
        return self

    def __ilshift__(self, other):
        self.__value__ <<= other
        return self

    def __imatmul__(self, other):
        self.__value__ @= other
        return self

    def __imul__(self, other):
        self.__value__ *= other
        return self

    def __index__(self):
        return self.__value__

    def __sub__(self, other):
        return self.__value__ - other

    def __instancecheck__(self, instance):
        return isinstance(instance, int)

    def __int__(self):
        return self.__value__

    def __invert__(self):
        return ~self.__value__

    def __ior__(self, other):
        return self.__value__ or other

    def __ipow__(self, other):
        self.__value__ **= other
        return self

    def __irshift__(self, other):
        self.__value__ >>= other
        return self

    def __isub__(self, other):
        self.__value__ -= other
        return self

    def __itruediv__(self, other):
        self.__value__ /= other
        return self

    def __ixor__(self, other):
        self.__value__ ^= other
        return self

    def __long__(self):
        return long(self.__value__)

    def __lshift__(self, other):
        return self.__value__ << other

    def __matmul__(self, other):
        return self.__value__ @ other

    def __mod__(self, other):
        return self.__value__ % other

    def __neg__(self):
        return -self.__value__

    def __oct__(self):
        return oct(self.__value__)

    def __or__(self, other):
        return self.__value__ or other

    def __pow__(self, power, modulo=None):
        return pow(self.__value__, power, modulo)

    def __pos__(self):
        return +self.__value__

    def __radd__(self, other):
        return other + self.__value__

    def __rand__(self, other):
        return other and self.__value__

    def __rdiv__(self, other):
        return other / self.__value__

    def __rdivmod__(self, other):
        return divmod(other, self.__value__)

    def __repr__(self):
        return repr(self.__value__)

    def __rfloordiv__(self, other):
        return other // self.__value__

    def __rlshift__(self, other):
        return other << self.__value__

    def __rmatmul__(self, other):
        return other @ self.__value__

    def __rmod__(self, other):
        return other % self.__value__

    def __rmul__(self, other):
        return other * self.__value__

    def __ror__(self, other):
        return other or self.__value__

    def __round__(self, n=None):
        return round(self.__value__, n)

    def __rpow__(self, other):
        return other ** self.__value__

    def __rrshift__(self, other):
        return other >> self.__value__

    def __rshift__(self, other):
        return self.__value__ >> other

    def __rsub__(self, other):
        return other - self.__value__

    def __rtruediv__(self, other):
        return other / self.__value__

    def __rxor__(self, other):
        return other ^ self.__value__

    def __truediv__(self, other):
        return self.__value__ / other

    def __trunc__(self):
        return math.trunc(self.__value__)

    def __unicode__(self):
        return unicode(self.__value__)

    def __and__(self, other):
        return self.__value__ and other

    def __floordiv__(self, other):
        return self.__value__ // other

    def __xor__(self, other):
        return self.__value__ ^ other

    def __str__(self):
        return str(self.__value__)

    def __iter__(self):
        for _i in np.arange(32):
            yield self[_i]

    @staticmethod
    def _check_key_value(key: int):
        if key > 31:
            raise ValueError("int are 32 bits long, %d given is too much" % key)
        if key < 0:
            raise ValueError("Negative keys not supported, %d given is too low" % key)

    def __reversed__(self):
        for _i in np.arange(31, 0, step=-1):
            yield self[_i]

    def __getitem__(self, key: int) -> int:
        """Returns the digit number *key* from the binary representation of *self* (little endian)

        Args:
            key (int):

        Returns:
            int:
        """
        EnhancedInt._check_key_value(key)
        return (self.__value__ >> key) & 1

    def __setitem__(self, key, value) -> None:
        """Changes the value of the *key*th bit of *self* (little endian)

        Args:
            key:
            value:

        Returns:

        """
        EnhancedInt._check_key_value(key)
        if value != 0 and value != 1:
            raise ValueError("Value must be 0 or 1, %d given" % value)
        if (not self[key]) and value:
            self.__value__ += 1 << key
            return None
        if self[key] and not value:
            self.__value__ -= 1 << key

    def __delitem__(self, key):
        """Removes the *key*th bit of *self*, the more significant bits are moved to te left (divided by 2) (little endian)

        Args:
            key:

        Returns:

        """
        EnhancedInt._check_key_value(key)
        _end_word = EnhancedInt(self.__value__)
        for _i in np.arange(key, 32):
            self[_i] = 0
        for _i in np.arange(key + 1):
            _end_word[_i] = 0
        self.__value__ += _end_word >> 1

    def insert(self, index: int, value: int):
        if value != 0 and value != 1:
            raise ValueError("Value must be 0 or 1, %d given" % value)
        if self.bit_length() == 32:
            raise OverflowError('Can\'t add a digit to a int already 32 bit long')
        for _i in np.arange(31, index, -1):
            self[_i] = self[_i - 1]
        self[index] = value

    def index(self, value: int) -> int:
        if value != 0 and value != 1:
            raise ValueError("Value must be 0 or 1, %d given" % value)
        for i in np.arange(32):
            if self[i] == value:
                return i
        return -1

    def count(self, i: int) -> int:
        if i != 0 and i != 1:
            raise ValueError("Value must be 0 or 1, %d given" % i)
        count = 0
        for index in np.arange(self.bit_length()):
            if self[index] == i:
                count += 1
        return count

    def bit_length(self):
        try:
            _last_1 = list(reversed(self)).index(1)
        except ValueError:
            _last_1 = 31
        # return 32 - list(reversed(self)).index(1)
        return 32 - _last_1


GenericMatrix = TypeVar('GenericMatrix', np.matrix, AbstractMatrix)

UsedFragment = Pi4Matrix
