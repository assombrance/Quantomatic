# coding=UTF-8
"""
Module focused on setting up the various data format needed. test
"""
import inspect
import math
from copy import deepcopy
from typing import List, TypeVar, Union, SupportsInt

import numpy as np


class PrettyStr:
    """
    Utility class redefining the string cast be be readable (returning all the attributes in the string)
    """
    def __str__(self) -> str:
        string = "<" + self.__class__.__name__ + ": "
        for atr in sorted(self.__dict__):
            string += str(self.__getattribute__(atr)) + " (" + atr + "), "
        string = string[:-2] + ">"
        return string


class AtrComparison:
    """
    Utility class redefining the __eq__ and the __hash__ magic methods so classes implementing this class are have their
    attributes compared instead of their identity (this would have been done using their id)
    """
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class Wire(PrettyStr, AtrComparison):
    """
    Used in the graph form representation.
    A wire is initially an input or an output of the graph (the name comes from the current name used in Quantomatic).
    In our implementation, a wire can also be a node for technical reasons.

    Attributes:
        name (str): name of the wire
    """
    def __init__(self, name: str) -> None:
        self.name = name


class Node(Wire):
    """
    Used in the graph form representation.
    A Node is inside the graph, it inherits from wire so the edge can link either two wires together, a wire and a node
    or two nodes together.

    Attributes:
        angle (float): angle of the node
        arity (int): arity of the node
        node_type (str): type of the node. As of now *(V2.0)*, the node types implemented are *X*, *Z* and *hadamard*
    """
    def __init__(self, name: str, angle: float = 0., node_type: str = 'Z', arity: int = 0) -> None:
        super().__init__(name)
        self.angle = angle
        self.arity = arity
        self.node_type = node_type


class Edge(PrettyStr, AtrComparison):
    """
    Used in the graph form representation.
    An edge links two wires together.

    Attributes:
        name (str): name of the edge
        n1 (Wire): first wire of the edge
        n2 (Wire): second wire of the edge
        label (str): label of the edge
    """
    def __init__(self, name: str, n1: Wire, n2: Wire, label: str = ""):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.label = label

    def __iter__(self):
        """
        This special method is used to make the edge iterable, this way, it is easier to access both wires at the same
        time.

        Yields:
            The next wire (between the two wires of the edge).
        """
        yield self.n1
        yield self.n2


class Graph:
    """
    Structure containing all the information of a graph in one place, nodes are theatrically unnecessary but are very
    useful to access directly (and we could think about situations where a graph is in a transition step and needs a
    disjunction between the nodes and the edges).
    """
    def __init__(self, nodes: List[Node]=None, edges: List[Edge]=None,
                 inputs: List[Wire]=None, outputs: List[Wire]=None):
        if nodes is None:
            nodes = []
        if edges is None:
            edges = []
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        self.nodes = nodes
        self.edges = edges
        self.inputs = inputs
        self.outputs = outputs

    def __bool__(self):
        return bool(self.outputs) or bool(self.nodes) or bool(self.edges) or bool(self.inputs)

    def augment(self, containing_graph: 'Graph') -> bool:
        """ Increase self by adding it's neighbours and the edges linking it to those neighbours, as well as edges
        linking neighbours between them. Returns *True* is the graph has been augmented, *False* otherwise

        Args:
            containing_graph (Graph): graph to pick neighbours from

        Returns:
            bool: *True* if the graph has been augmented, *False* otherwise
        """
        neighbours = self.neighbours(containing_graph)
        if neighbours.inputs + neighbours.outputs + neighbours.nodes:
            self.outputs += neighbours.outputs
            self.inputs += neighbours.inputs
            self.nodes += neighbours.nodes
            self.edges += neighbours.edges
            return True
        else:
            return False

    def neighbours(self, containing_graph: 'Graph') -> 'Graph':
        """ Returns a partial graph (not all edges' ends are necessarily in the nodes or in the I/O) containing all the
        neighbours wires from *containing_graph* as well as the edges between those neighbours and the edges between the
        neighbours and the wire from *self*

        Args:
            containing_graph (Graph): graph to pick the neighbours from

        Returns:
            Graph: graph containing the neighbours of *self* and the proper edges
        """
        neighbours = Graph()
        for edge in containing_graph.edges:
            start_wires = self.outputs + self.inputs + self.nodes
            intersection = [wire for wire in edge if wire in start_wires]
            if len(intersection) == 1:
                neighbour = list(set(edge) - set(intersection))[0]
                if neighbour in containing_graph.outputs and neighbour not in neighbours.outputs:
                    neighbours.outputs.append(neighbour)
                if neighbour in containing_graph.inputs and neighbour not in neighbours.inputs:
                    neighbours.inputs.append(neighbour)
                if neighbour in containing_graph.nodes and neighbour not in neighbours.nodes:
                    neighbours.nodes.append(neighbour)
                neighbours.edges.append(edge)
                temp_input_edge_wire = [wire for wire in containing_graph.inputs if wire.name == edge.name]
                if temp_input_edge_wire and temp_input_edge_wire[0] not in neighbours.inputs + self.inputs:
                    neighbours.inputs.append(temp_input_edge_wire[0])
                    neighbours.edges.append(edge)
                temp_output_edge_wire = [wire for wire in containing_graph.outputs if wire.name == edge.name]
                if temp_output_edge_wire and temp_output_edge_wire[0] not in neighbours.outputs + self.outputs:
                    neighbours.outputs.append(temp_output_edge_wire[0])
                    neighbours.edges.append(edge)
        for edge in containing_graph.edges:
            intersection = set(neighbours.outputs + neighbours.inputs + neighbours.nodes)\
                .intersection(edge)
            if len(intersection) == 2 or (edge.n1 in intersection and edge.n1 == edge.n2):
                neighbours.edges.append(edge)
        return neighbours

    def __add__(self, other: 'Graph') -> 'Graph':
        if not isinstance(other, Graph):
            raise TypeError('other should be a Graph')
        result = Graph()
        result.outputs = self.outputs + other.outputs
        result.inputs = self.inputs + other.inputs
        result.nodes = self.nodes + other.nodes
        result.edges = self.edges + other.edges
        return result

    def __iadd__(self, other: 'Graph'):
        if not isinstance(other, Graph):
            raise TypeError('other should be a Graph')
        self.outputs += other.outputs
        self.inputs += other.inputs
        self.nodes += other.nodes
        self.edges += other.edges
        return self

    def __sub__(self, other: 'Graph') -> 'Graph':
        if not isinstance(other, Graph):
            raise TypeError('other should be a Graph')
        result = Graph()
        result.outputs = [outputWire for outputWire in self.outputs if outputWire not in other.outputs]
        result.inputs = [inputWire for inputWire in self.inputs if inputWire not in other.inputs]
        result.nodes = [node for node in self.nodes if node not in other.nodes]
        result.edges = [edge for edge in self.edges if edge not in other.edges]
        return result

    def __eq__(self, other):
        if not isinstance(other, Graph):
            return False
        inputs_diff = set(self.inputs).symmetric_difference(other.inputs)
        outputs_diff = set(self.outputs).symmetric_difference(other.outputs)
        nodes_diff = set(self.nodes).symmetric_difference(other.nodes)
        edges_diff = set(self.edges).symmetric_difference(other.edges)
        return not (inputs_diff or outputs_diff or nodes_diff or edges_diff)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        string = "\n<" + self.__class__.__name__ + ": \n"
        if self.inputs:
            string += "["
            for input_wire in self.inputs:
                string += str(input_wire) + ", "
            string = string[:-2] + "] (inputs), \n"
        else:
            string += "[] (inputs), \n"
        if self.outputs:
            string += "["
            for output_wire in self.outputs:
                string += str(output_wire) + ", "
            string = string[:-2] + "] (outputs), \n"
        else:
            string += "[] (outputs), \n"
        if self.nodes:
            string += "["
            for node in self.nodes:
                string += str(node) + ", "
            string = string[:-2] + "] (nodes), \n"
        else:
            string += "[] (nodes), \n"
        if self.edges:
            string += "["
            for edge in self.edges:
                string += str(edge) + ", "
            string = string[:-2] + "] (edges)>\n"
        else:
            string += "[] (edges)>\n"
        return string


class ConnectionPoint(PrettyStr, AtrComparison):
    """
    Used in the matrix form representation.
    In this software, at one level of iteration given, only two matrices are considered. This class describes a point of
    one of those two matrices, it is a combination of :

    Attributes:
        is_matrix_2 (bool): if false, the point is on the first matrix, otherwise, it is on the second one. This
            paradigm has been chosen because of the falsy value of 0 (assigning this parameter to 0 would mean the
            connection point would be on the first matrix and assigning it to one would be on the second matrix)
        is_out (bool): if false, is an input, otherwise is an output. This choice has been lead by the same reasoning as
            for the is_matrix_2 attribute.
        index (int): index of the edge connection. Beware : this index is not between 0 and the size of the matrix in
            the related direction but between 0 and the bit length of this same size.
    """
    def __init__(self, is_matrix_2: bool=False, is_out: bool=False, index: int=0) -> None:
        self.is_matrix_2 = is_matrix_2
        self.is_out = is_out
        self.index = index


class InterMatrixLink(PrettyStr, AtrComparison):
    """
    Used in the matrix form representation.
    An InterMatrixLink links two ConnectionPoints. Those do not have to be on different matrices, or even on different
    sides of a matrix.

    Attributes:
        point1 (ConnectionPoint): first connection point
        point2 (ConnectionPoint): second connection point
    """
    def __init__(self, point1: ConnectionPoint, point2: ConnectionPoint) -> None:
        self.point1 = point1
        self.point2 = point2


class AbstractMatrix(AtrComparison, np.matrix):
    """
    Alternative to numpy matrix in the algorithm, this class is entirely abstract (it cannot be constructed) and
    specifies all the methods needed to implement a new kind of matrix.
    """
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
        """
        a.dot(b, out=None)

            Dot product of two arrays.

            Refer to `numpy.dot` for full documentation.

            See Also
            --------
            numpy.dot : equivalent function

            Examples
            --------
            >>> a = np.eye(2)
            >>> b = np.ones((2, 2)) * 2
            >>> a.dot(b)
            array([[ 2.,  2.],
                   [ 2.,  2.]])

            This array method can be conveniently chained:

            >>> a.dot(b).dot(b)
            array([[ 8.,  8.],
                   [ 8.,  8.]])
        """
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
    def node_to_matrix(cls, node: Node, in_number: int, out_number: int) -> 'GenericMatrix':
        """
        Converts a node to a matrix in the correct fragment.

        Args:
            node(Node): the node to be converted
            in_number (int): number of inputs to the node
            out_number (int): number of outputs to the node

        Returns:
            GenericMatrix: The matrix corresponding to the node.
        """
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (cls.__name__, inspect.stack()[0][3]))

    @property
    def shape(self):
        """Returns the shape of the matrix

        Returns:
            (int,int): Shape of the matrix
        """
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    @property
    def size(self):
        """Returns the size of the matrix

        Returns:
            int: Size of the matrix
        """
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def tensor_product(self, b) -> 'GenericMatrix':
        """
        Basic tensor product of *self* and *b*.

        Examples:
            A.tensor_product(B)=

            =========  =====  ==========
            A(0,0)*B   ...    A(0,-1)*B
              ...               ...
            A(-1,0)*B  ...    A(-1,-1)*B
            =========  =====  ==========

        Args:
            b (GenericMatrix): matrix to be multiplied (in the tensor way) to *self*

        Returns:
            GenericMatrix: tensor product of *self* and *b*
        """
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def tensor_power(self, power: int) -> 'GenericMatrix':
        """*power*th power of *self* in the tensor way (understand use tensor product instead of classical product)

        Args:
            power (int): power to be elevated to

        Returns:
            GenericMatrix: the computed power
        """
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
        """
        Test whether any array element along a given axis evaluates to True.

        Refer to `numpy.any` for full documentation.

        Parameters
        ----------
        axis : int, optional
            Axis along which logical OR is performed
        out : ndarray, optional
            Output to existing array instead of creating new one, must have
            same shape as expected output

        Returns
        -------
            any : bool, ndarray
                Returns a single bool if `axis` is ``None``; otherwise,
                returns `ndarray`

        """
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def all(self, axis=None, out=None):
        """
        Test whether all matrix elements along a given axis evaluate to True.

        Parameters
        ----------
        See `numpy.all` for complete descriptions

        See Also
        --------
        numpy.all

        Notes
        -----
        This is the same as `ndarray.all`, but it returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> y = x[0]; y
        matrix([[0, 1, 2, 3]])
        >>> (x == y)
        matrix([[ True,  True,  True,  True],
                [False, False, False, False],
                [False, False, False, False]])
        >>> (x == y).all()
        False
        >>> (x == y).all(0)
        matrix([[False, False, False, False]])
        >>> (x == y).all(1)
        matrix([[ True],
                [False],
                [False]])

        """
        raise NotImplementedError("Class %s doesn't implement %s()" %
                                  (self.__class__.__name__, inspect.stack()[0][3]))

    def reshape(self, shape, *shapes, order='C'):
        """
        a.reshape(shape, order='C')

            Returns an array containing the same data with a new shape.

            Refer to `numpy.reshape` for full documentation.

            See Also
            --------
            numpy.reshape : equivalent function

            Notes
            -----
            Unlike the free function `numpy.reshape`, this method on `ndarray` allows
            the elements of the shape parameter to be passed in as separate arguments.
            For example, ``a.reshape(10, 11)`` is equivalent to
            ``a.reshape((10, 11))``.
        """
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
    """
    Implementation of AbstractMatrix, in the pi/4 fragment.
    In this fragment, a matrix is represented by 4 matrices, multiplied respectively by exp(k\*i\*pi) with k from 0 to 4
    A power of 2 is also used to avoid (a little bit) number from exploding.
    """
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
        """Returns the shape of the matrix

        Returns:
            (int,int): Shape of the matrix
        """
        return self.m0.shape

    @property
    def size(self):
        """Returns the size of the matrix

        Returns:
            int: Size of the matrix
        """
        return self.m0.size

    @classmethod
    def node_to_matrix(cls, node: Node, in_number: int, out_number: int) -> 'Pi4Matrix':
        """Returns the matrix in the Pi4 fragment corresponding to the semantic of the given *node*.

        Args:
            node(Node): the node to be converted
            in_number (int): number of inputs to the node
            out_number (int): number of outputs to the node

        Returns:

        """
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
            result = Pi4Matrix(null_base, base, null_base, -base, 1)
        elif node.node_type == 'not-triangle':
            if in_number + out_number != 2:
                raise ValueError('Hadamard gate is only known for node with an arity of 2')
            result = Pi4Matrix(np.ones((1 << out_number, 1 << in_number)))
            result[-1, -1] = Pi4Matrix(0)
        elif node.node_type == 'Z' or node.node_type == 'X':
            position = node.angle * fragment
            while position < 0:
                position += 2*fragment
            if position in np.arange(2*fragment):
                null_base = np.zeros((1 << out_number, 1 << in_number))
                base = deepcopy(null_base)
                base[0, 0] = 1
                matrices = [base]
                for _ in np.arange(fragment):
                    matrices.append(deepcopy(null_base))
                if position < fragment:
                    matrices[int(position)][-1, -1] += 1
                else:
                    matrices[int(position) - fragment][-1, -1] += -1
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
        """To have a unique representation for a matrix described by Pi4Matrix, the dividing 2's power must be as great
        as possible.

        To achieve this, this methods ensure that at least one coefficient of one of the matrices is odd, and increase
        the 2's power accordingly.
        """
        smallest_2_power = 32
        non_zero_coefficient_found = False
        height, width = self.shape
        for i in np.arange(height):
            for j in np.arange(width):
                x = EnhancedInt(abs(self.m0[i, j]))
                if x.index(1) >= 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
                x = EnhancedInt(abs(self.m1[i, j]))
                if x.index(1) >= 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
                x = EnhancedInt(abs(self.m2[i, j]))
                if x.index(1) >= 0:
                    non_zero_coefficient_found = True
                    if x.index(1) < smallest_2_power:
                        smallest_2_power = x.index(1)
                x = EnhancedInt(abs(self.m3[i, j]))
                if x.index(1) >= 0:
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
            raise ValueError('Tensor power not defined for a negative power')
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
        """
        a.dot(b, out=None)

            Dot product of two arrays.

            Refer to `numpy.dot` for full documentation.

            See Also
            --------
            numpy.dot : equivalent function

            Examples
            --------
            >>> a = np.eye(2)
            >>> b = np.ones((2, 2)) * 2
            >>> a.dot(b)
            array([[ 2.,  2.],
                   [ 2.,  2.]])

            This array method can be conveniently chained:

            >>> a.dot(b).dot(b)
            array([[ 8.,  8.],
                   [ 8.,  8.]])
        """
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
        """
        Test whether any array element along a given axis evaluates to True.

        Refer to `numpy.any` for full documentation.

        Parameters
        ----------
        axis : int, optional
            Axis along which logical OR is performed
        out : ndarray, optional
            Output to existing array instead of creating new one, must have
            same shape as expected output

        Returns
        -------
            any : bool, ndarray
                Returns a single bool if `axis` is ``None``; otherwise,
                returns `ndarray`

        """
        return self.m0.any(axis, out) or self.m1.any(axis, out) or self.m2.any(axis, out) or self.m3.any(axis, out)

    def all(self, axis=None, out=None):
        """
        Test whether all matrix elements along a given axis evaluate to True.

        Parameters
        ----------
        See `numpy.all` for complete descriptions

        See Also
        --------
        numpy.all

        Notes
        -----
        This is the same as `ndarray.all`, but it returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> y = x[0]; y
        matrix([[0, 1, 2, 3]])
        >>> (x == y)
        matrix([[ True,  True,  True,  True],
                [False, False, False, False],
                [False, False, False, False]])
        >>> (x == y).all()
        False
        >>> (x == y).all(0)
        matrix([[False, False, False, False]])
        >>> (x == y).all(1)
        matrix([[ True],
                [False],
                [False]])

        """
        return self.m0.any(axis, out) and self.m1.any(axis, out) and self.m2.any(axis, out) and self.m3.any(axis, out)

    def reshape(self, shape, *shapes, order='C'):
        """
        a.reshape(shape, order='C')

            Returns an array containing the same data with a new shape.

            Refer to `numpy.reshape` for full documentation.

            See Also
            --------
            numpy.reshape : equivalent function

            Notes
            -----
            Unlike the free function `numpy.reshape`, this method on `ndarray` allows
            the elements of the shape parameter to be passed in as separate arguments.
            For example, ``a.reshape(10, 11)`` is equivalent to
            ``a.reshape((10, 11))``.
        """
        _m0 = self.m0.reshape(shape, *shapes)
        _m1 = self.m1.reshape(shape, *shapes)
        _m2 = self.m2.reshape(shape, *shapes)
        _m3 = self.m3.reshape(shape, *shapes)
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
            key: key to set
            value (Pi4Matrix): value to set
        """
        if not isinstance(value, Pi4Matrix):
            return TypeError("Value should be a matrix")
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
    """
    Does everything a int can do, but can also be addressed bitwise (you can read, write, add and delete a bit at a
    given position)

    Bit representation is in little endian : the lowest indexes corresponding to the least significant bits
    """
    def __init__(self, x: Union[str, bytes, SupportsInt]) -> None:
        int.__init__(int(x))
        self.__value__ = int(x)

    @staticmethod
    def from_list(x: List[int]):
        """Turns a list of bits into an EnhancedInt

        Args:
            x (List[int]): list of bits

        Returns:
            EnhancedInt: The EnhancedInt corresponding to the bit list
        """
        _value = 0
        if isinstance(x, List):
            if len(x) > 32:
                raise ValueError("For now, only 32 bit ints are implemented, list too long (%d)" % len(x))
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

    def __lshift__(self, other):
        return self.__value__ << other

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
            key (int): bit number to be returned

        Returns:
            int: value of the bit addressed
        """
        EnhancedInt._check_key_value(key)
        return (self.__value__ >> key) & 1

    def __setitem__(self, key: int, value: int) -> None:
        """Changes the value of the *key*th bit of *self* (little endian)

        Args:
            key (int): index of bit to be modified
            value (int): bit value (must be 0 or 1)
        """
        EnhancedInt._check_key_value(key)
        if value != 0 and value != 1:
            raise ValueError("Value must be 0 or 1, %d given" % value)
        if (not self[key]) and value:
            self.__value__ += 1 << key
            return None
        if self[key] and not value:
            self.__value__ -= 1 << key

    def __delitem__(self, key: int):
        """Removes the *key*th bit of *self*, the more significant bits are moved to te left (divided by 2) (little endian)

        Args:
            key (int): index of bit to be deleted
        """
        EnhancedInt._check_key_value(key)
        _end_word = EnhancedInt(self.__value__)
        for _i in np.arange(key, 32):
            self[_i] = 0
        for _i in np.arange(key + 1):
            _end_word[_i] = 0
        self.__value__ += _end_word >> 1

    def insert(self, index: int, value: int):
        """Insert *value* in *self* at index *index*.

        Args:
            index (int): insertion index
            value (int): bit value (must be in [0, 1]
        """
        if value != 0 and value != 1:
            raise ValueError("Value must be 0 or 1, %d given" % value)
        if self.bit_length() == 32:
            raise OverflowError('Can\'t add a digit to a int already 32 bit long')
        for _i in np.arange(31, index, -1):
            self[_i] = self[_i - 1]
        self[index] = value

    def index(self, value: int) -> int:
        """Returns the index of the first *value* found in *self*. *value* is the value of a bit, hence, it must be 0 or 1.
        If *value* is not found, will return -1

        Args:
            value (int): bit value to be searched

        Returns:
            int: index of *value*
        """
        if value != 0 and value != 1:
            raise ValueError("Value must be 0 or 1, %d given" % value)
        for i in np.arange(32):
            if self[i] == value:
                return i
        return -1

    def count(self, i: int) -> int:
        """Returns the number of *i* in *self*. *i* is the value of a bit, hence, it must be 0 or 1.

        Examples:
            >>> EnhancedInt(5).count(1)
            2
            # bit representation of 5 : 0...0101

        Args:
            i (int): bit value to be counted

        Returns:
            int: number of *i* found in *self*
        """
        if i != 0 and i != 1:
            raise ValueError("Value must be 0 or 1, %d given" % i)
        count = 0
        for index in np.arange(self.bit_length()):
            if self[index] == i:
                count += 1
        return count

    def bit_length(self):
        """Returns the bit length of *self*. If *self* == 0, then will return 1

        Returns:
            int: bit size of *self*
        """
        try:
            _last_1 = list(reversed(self)).index(1)
        except ValueError:
            _last_1 = 31
        return 32 - _last_1


GenericMatrix = TypeVar('GenericMatrix', np.matrix, AbstractMatrix)

UsedFragment = Pi4Matrix
