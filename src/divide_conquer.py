import inspect
import math
import sys
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


class Node(PrettyStr, AtrComparison):
    def __init__(self, name: str, angle: str = '0', arity: int = 0) -> None:
        self.name = name
        self.angle = angle
        self.arity = arity


class ConnectionPoint(PrettyStr, AtrComparison):
    def __init__(self, matrix=False, is_out=False, index=0) -> None:
        self.matrix = matrix
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


class Pi4Matrix(AbstractMatrix):
    @property
    def shape(self):
        return self.m0.shape

    @property
    def size(self):
        return self.m0.size

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

    def dot(self, b, out=None):
        _pi_4_b = Pi4Matrix(b)
        if _pi_4_b.shape[0] == self.shape[1]:
            _z = _pi_4_b.z + self.z
            _m0 = self.m0.dot(_pi_4_b.m0) + self.m1.dot(_pi_4_b.m3) + \
                self.m2.dot(_pi_4_b.m2) + self.m3.dot(_pi_4_b.m1)
            _m1 = self.m0.dot(_pi_4_b.m1) + self.m1.dot(_pi_4_b.m0) + \
                self.m2.dot(_pi_4_b.m3) + self.m3.dot(_pi_4_b.m2)
            _m2 = self.m0.dot(_pi_4_b.m2) + self.m1.dot(_pi_4_b.m1) + \
                self.m2.dot(_pi_4_b.m0) + self.m3.dot(_pi_4_b.m3)
            _m3 = self.m0.dot(_pi_4_b.m3) + self.m1.dot(_pi_4_b.m2) + \
                self.m2.dot(_pi_4_b.m1) + self.m3.dot(_pi_4_b.m0)
            return Pi4Matrix(_m0, _m1, _m2, _m3, _z)
        raise ValueError('Size mismatch')

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
        raise ValueError('Size mismatch')

    def __iadd__(self, y):
        return super().__iadd__(y)

    def __radd__(self, y):
        return super().__radd__(y)

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
        raise ValueError('Size mismatch')

    def __isub__(self, y):
        return super().__isub__(y)

    def __rsub__(self, y):
        return super().__rsub__(y)

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
        return super().any(axis, out)

    def all(self, axis=None, out=None):
        return super().all(axis, out)

    def reshape(self, shape, *shapes, order='C'):
        _m0 = self.m0.reshape(shape, *shapes, order)
        _m1 = self.m1.reshape(shape, *shapes, order)
        _m2 = self.m2.reshape(shape, *shapes, order)
        _m3 = self.m3.reshape(shape, *shapes, order)
        return Pi4Matrix(_m0, _m1, _m2, _m3, self.z)

    def __array__(self, dtype=None):
        super().__array__(dtype)

    def __format__(self, *args, **kwargs):
        super().__format__(*args, **kwargs)

    def __imatmul__(self, *args, **kwargs):
        super().__imatmul__(*args, **kwargs)

    def __imod__(self, y):
        return super().__imod__(y)

    def __matmul__(self, *args, **kwargs):
        super().__matmul__(*args, **kwargs)

    def __neg__(self, *args, **kwargs):
        return super().__neg__(*args, **kwargs)

    def __pos__(self, *args, **kwargs):
        return super().__pos__(*args, **kwargs)

    def __rmatmul__(self, *args, **kwargs):
        super().__rmatmul__(*args, **kwargs)

    def __sizeof__(self, *args, **kwargs):
        super().__sizeof__(*args, **kwargs)

    def __eq__(self, other):
        pi_4_other = Pi4Matrix(other)
        _m0 = (self.m0 * self.z - pi_4_other.m0 * pi_4_other.z).all()
        _m1 = (self.m1 * self.z - pi_4_other.m1 * pi_4_other.z).all()
        _m2 = (self.m2 * self.z - pi_4_other.m2 * pi_4_other.z).all()
        _m3 = (self.m3 * self.z - pi_4_other.m3 * pi_4_other.z).all()
        return _m0 and _m1 and _m2 and _m3

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


def fusion_matrices(m1: GenericMatrix, m2: GenericMatrix, inputs: List[ConnectionPoint],
                    outputs: List[ConnectionPoint], links: List[InterMatrixLink]) -> GenericMatrix:
    """ Main component of the V2 of the algorithm uses clever algorithms to lower the complexity.
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
    # use of while instead of for to be able to remove values inplace
    iteration_index = 0
    while iteration_index < len(links):
        link = links[iteration_index]
        removed_points = []  # type: List[ConnectionPoint]
        # we don't need to memorize the points we'll add because we will always add them last
        if link.point1.matrix == link.point2.matrix:
            if link.point1.is_out != link.point2.is_out:
                # set the point2 to the same side of the matrix as point1
                # and remove the resulting cap or cup
                if link.point1.matrix:
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
                if link.point1.matrix:
                    m2 = remove_cup(m2, link.point1.index, link.point2.index)
                else:
                    m1 = remove_cup(m1, link.point1.index, link.point2.index)
            else:
                # remove cap
                if link.point1.matrix:
                    m2 = remove_cap(m2, link.point1.index, link.point2.index)
                else:
                    m1 = remove_cap(m1, link.point1.index, link.point2.index)
            removed_points.append(link.point1)
            removed_points.append(link.point2)
        elif not link.point1.matrix and not link.point1.is_out:
            # input from matrix 1 connected to matrix 2
            # switch point1 to be an output
            m1 = input_to_output(m1, link.point1.index)
            removed_points.append(deepcopy(link.point1))
            link.point1.is_out = True
            link.point1.index = -1
        elif not link.point2.matrix and not link.point2.is_out:
            # input from matrix 1 connected to matrix 2
            # switch point2 to be an output
            m1 = input_to_output(m1, link.point2.index)
            removed_points.append(deepcopy(link.point2))
            link.point2.is_out = True
            link.point2.index = -1
        elif link.point2.matrix and link.point2.is_out:
            # output from matrix 2 connected to matrix 1
            # switch point2 to be an input
            m2 = output_to_input(m2, link.point2.index)
            removed_points.append(deepcopy(link.point2))
            link.point2.is_out = False
            link.point2.index = -1
        elif link.point1.matrix and link.point1.is_out:
            # output from matrix 2 connected to matrix 1
            # switch point1 to be an input
            m2 = output_to_input(m2, link.point1.index)
            removed_points.append(deepcopy(link.point1))
            link.point1.is_out = False
            link.point1.index = -1  # TODO: better later ? problem with reference ?
        # move the indexes of the other points to the correct new positions
        if link.point1 in removed_points and link.point2 in removed_points:
            del links[iteration_index]
        else:
            iteration_index += 1
        if link.point1.index == -1:
            if not link.point1.matrix:
                if not link.point1.is_out:
                    link.point1.index = EnhancedInt.from_list(m1.shape[1] * [1]) - 1  # TODO faux !
                else:
                    link.point1.index = EnhancedInt.from_list(m1.shape[0] * [1]) - 1  # TODO faux !
            else:
                if not link.point1.is_out:
                    link.point1.index = EnhancedInt.from_list(m2.shape[1] * [1]) - 1  # TODO faux !
                else:
                    link.point1.index = EnhancedInt.from_list(m2.shape[0] * [1]) - 1  # TODO faux !
        for removed_point in removed_points:
            for link in links:
                if link.point1.matrix == removed_point.matrix and link.point1.is_out == removed_point.is_out \
                        and link.point1.index > removed_point.index:
                    link.point1.index -= 1
                if link.point2.matrix == removed_point.matrix and link.point2.is_out == removed_point.is_out \
                        and link.point2.index > removed_point.index:
                    link.point2.index -= 1
            for input_point in inputs:
                if input_point.matrix == removed_point.matrix and input_point.is_out == removed_point.is_out \
                        and input_point.index > removed_point.index:
                    input_point.index -= 1
            for output_point in outputs:
                if output_point.matrix == removed_point.matrix and output_point.is_out == removed_point.is_out \
                        and output_point.index > removed_point.index:
                    output_point.index -= 1
    for input_point in inputs:
        if input_point.is_out:
            # change this input to be an input for the matrix
            if not input_point.matrix:
                m1 = output_to_input(m1, input_point.index)
                input_point.is_out = False
                input_point.index = EnhancedInt.from_list(m1.shape[1] * [1]) - 1  # TODO faux !
            pass
    for output_point in outputs:
        if not output_point.is_out:
            # change this output to be an output for the matrix
            pass

    inputs_m1 = [input_point for input_point in inputs if not input_point.matrix]
    inputs_m1_sorted = sorted(inputs_m1, key=lambda input_m1: input_m1.index)

    m1_m2_links = [link for link in links if link.point1.matrix != link.point2.matrix]
    covering = len(m1_m2_links)

    def links_sorter(_link: InterMatrixLink, _matrix: bool):
        if _link.point1.matrix == _matrix:
            return _link.point1.index
        else:
            return _link.point2.index

    m1_m2_links_m1_sorted = sorted(m1_m2_links, key=lambda _link: links_sorter(_link, False))

    outputs_m1_to_m2 = [link.point1 if link.point1.matrix else link.point2 for link in m1_m2_links_m1_sorted]
    outputs_m1_to_output = [output_point for output_point in outputs if not output_point.matrix]

    outputs_m1 = outputs_m1_to_output + outputs_m1_to_m2
    outputs_m1_sorted = sorted(outputs_m1, key=lambda output_m1: output_m1.index)

    m1_ordered = order_matrix(m1, inputs_m1, inputs_m1_sorted, outputs_m1_sorted, outputs_m1)

    inputs_m2_from_inputs = [input_point for input_point in inputs if input_point.matrix]
    inputs_m2_from_m1 = [link.point1 if not link.point1.matrix else link.point2 for link in m1_m2_links_m1_sorted]

    inputs_m2 = inputs_m2_from_m1 + inputs_m2_from_inputs
    inputs_m2_sorted = sorted(inputs_m2, key=lambda input_m2: input_m2.index)

    outputs_m2 = [output_point for output_point in outputs if output_point.matrix]
    outputs_m2_sorted = sorted(outputs_m2, key=lambda output_m2: output_m2.index)

    m2_ordered = order_matrix(m2, inputs_m2, inputs_m2_sorted, outputs_m2, outputs_m2_sorted)

    result = twisted_multiplication(m1_ordered, m2_ordered, covering)

    group_inputs = inputs_m1 + inputs_m2_from_inputs
    assert group_inputs == inputs
    group_inputs_sorted = sorted(group_inputs, key=lambda input_group: input_group.index)

    group_outputs = outputs_m1_to_output + outputs_m1
    assert group_outputs == outputs
    group_outputs_sorted = sorted(group_outputs, key=lambda output_group: output_group.index)

    result_ordered = order_matrix(result, group_inputs, group_inputs_sorted, group_outputs, group_outputs_sorted)

    return result_ordered


def remove_cup(m: GenericMatrix, index1: int, index2: int) -> GenericMatrix:
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
    """

    Args:
        m (GenericMatrix):
        input_index (int):
        output_index (int):

    Returns:
        GenericMatrix:
    """
    _height, _width = m.shape
    _input_base_size = _width.bit_length() - 1
    _new_input_base_size = _input_base_size - 1
    _output_base_size = _height.bit_length() - 1
    _new_output_base_size = _output_base_size + 1
    if input_index not in np.arange(_input_base_size):
        raise ValueError("Index out of bound, m input base size : %d and index1 given : %d" % (_input_base_size,
                                                                                               input_index))
    if output_index not in list(np.arange(_new_output_base_size)) + list(np.arange(-_new_output_base_size + 1, 0)):
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
        _i.insert(out_index, _enhanced_j[in_index])
        del _j[in_index]
        return _i, _j

    _matrix = UsedFragment(m)
    for _i in np.arange(2 ** _output_base_size):
        _enhanced_i = EnhancedInt(_i)
        for _j in np.arange(2 ** _input_base_size):
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
    """

    Args:
        m (GenericMatrix):
        input_index (int):
        output_index (int):

    Returns:
        GenericMatrix:
    """
    _height, _width = m.shape
    _input_base_size = _width.bit_length() - 1
    _new_input_base_size = _input_base_size + 1
    _output_base_size = _height.bit_length() - 1
    _new_output_base_size = _output_base_size - 1
    if input_index not in list(np.arange(_new_input_base_size)) + list(np.arange(-_new_input_base_size + 1, 0)):
        raise ValueError("Index out of bound, m input base size : %d and index1 given : %d" % (_input_base_size,
                                                                                               input_index))
    if output_index not in np.arange(_output_base_size):
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
            in_index: index of the destination
            out_index: index of the output to be taken

        Returns:

        """

        _j.insert(in_index, _i[out_index])
        del _i[out_index]
        return _i, _j

    _matrix = UsedFragment(m)
    for _i in np.arange(2 ** _output_base_size):
        _enhanced_i = EnhancedInt(_i)
        for _j in np.arange(2 ** _input_base_size):
            _enhanced_j = EnhancedInt(_j)
            if input_index < 0:
                _input_index = _new_input_base_size + input_index
            else:
                _input_index = input_index
            # the following operations with the _input_base_size and the _new_output_base_size are needed because
            # _transformation treats the numbers in little endian and the wires order is given in big endian
            _position = _transformation(_enhanced_i, _enhanced_j, _new_input_base_size - 1 - _input_index,
                                        _output_base_size - 1 - output_index)
            _result[_position] = _matrix[_i, _j]
    return _result


def order_matrix(m: GenericMatrix, input_order: list, matrix_input_order: list,
                 output_order: list, matrix_output_order: list) -> GenericMatrix:
    """

    Args:
        m (GenericMatrix):
        input_order (list):
        matrix_input_order (list):
        output_order (list):
        matrix_output_order (list):

    Returns:
        GenericMatrix:
    """
    if len(input_order) != len(matrix_input_order):
        raise ValueError('input_order and matrix_input_order length differ')
    if len(output_order) != len(matrix_output_order):
        raise ValueError('output_order and matrix_output_order length differ')
    _height, _width = m.shape
    _input_base_size = _width.bit_length() - 1
    _output_base_size = _height.bit_length() - 1
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
    """

    Args:
        m1 (GenericMatrix):
        m2 (GenericMatrix):
        covering (int):

    Returns:
        GenericMatrix
    """
    list_m1 = []  # type: List[UsedFragment]
    list_m2 = []  # type: List[UsedFragment]
    height_m1, width_m1 = EnhancedInt(m1.shape[0]), EnhancedInt(m1.shape[1])
    height_m2, width_m2 = EnhancedInt(m1.shape[0]), EnhancedInt(m1.shape[1])
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
            result[i * height_m1: (i + 1) * height_m1, j::width_m2 >> covering] = list_m12[i][j]
    return result


am = Pi4Matrix([[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 6, 0],
                [0, 0, 0, 0, 0, 0, 0, 7]])

# am = Pi4Matrix([[0, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 2, 0],
#                 [0, 0, 0, 3]])

# am = Pi4Matrix([[0,   1,  2,  3],
#                 [10, 11, 12, 13],
#                 [20, 21, 22, 23],
#                 [30, 31, 32, 33]])

# am = Pi4Matrix([[1, 2], [3, 4]])

# print(order_matrix(am, [0, 2, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2]))

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)
print(twisted_multiplication(am, am, 1))
