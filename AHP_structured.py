from __future__ import division
import numpy as np


class Compare(object):
    """
    This class computes the priority vector and consistency ratio of a positive
    reciprocal matrix. The 'weights' property contains the priority vector as a dictionary
    whose keys are criteria and whose values are the criteria's weights.
    The 'consistency_ratio' property contains the computed consistency ratio of the input matrix as a float.
    The 'remainder' property contains the difference (as a numpy array) between the final two
    eigenvectors used by the compute_priority_vector function.
    :param name: string, the name of the Compare object; if the object has a parent,
        this name MUST be included as a criterion of its parent
    :param matrix: numpy matrix, the matrix from which to derive the priority vector
    :param criteria: list of strings, the criteria of the matrix, listed in the same left-to-right
        order as their corresponding values in the input matrix
    :param precision: integer, number of decimal places of precision to compute both the priority
        vector and the consistency ratio; default is 4
    :param comp_type: string, the comparison type of the values in the input matrix, being either
        qualitative or quantitative; valid input: 'quant', 'qual'; default is 'qual'
    :param iters: integer, number of iterations before the compute_eigenvector function stops;
        default is 100
    :param random_index: string, the random index estimates used to compute the consistency ratio;
        valid input: 'dd', 'saaty'; default is 'dd'; see the compute_consistency_ratio function for more
        information regarding the different estimates
    """

    def __init__(self, name=None, matrix=None, criteria=None,
                 precision=4, comp_type='qual', iters=100, random_index='dd'):
        self.name = name
        self.matrix = None
        self.criteria = criteria
        self.shape = None
        self.type = comp_type
        self.precision = precision
        self.iterations = iters
        self.random_index = random_index.lower()
        self.priority_vector = None
        self.remainder = None
        self.consistency_ratio = None
        self.weights = None

        try:
            matrix = self.convert(matrix)
        except AttributeError:
            pass

        print matrix

        self.check_input(matrix)
        # self.matrix = matrix
        # self.shape = self.matrix.shape[0]
        self.compute()

    @staticmethod
    def convert(matrix_str):
        """
        Converts a string of form '1, 2; 3, 4' (or '1 2; 3 4') into a numpy matrix.
        :returns numpy matrix
        """

        matrix_1 = []
        try:
            for x in matrix_str.replace(',', ' ').split(';'):
                matrix_1.append([eval(y, {'__builtin__': None}, {}) for y in x.split()])
            dimension = len(matrix_1[0]) + 1
            matrix_2 = np.ones((dimension, dimension))
            for x, i in enumerate(matrix_1):
                for y, j in enumerate(i):
                    matrix_2.itemset((x, x + y + 1), j)
                    matrix_2.itemset((x + y + 1, x), 1 / j)
        except IndexError:
            return matrix_1
        except (NameError, SyntaxError):
            raise AHPException('Input contains invalid values')
        except ValueError:
            raise AHPException('Input not in numpy form "1 2; 3 4" or "1, 2; 3, 4"')
        return matrix_2

    def check_input(self, input_matrix):
        """
        Tests whether the input matrix of the Compare object can be cast as a matrix,
        and whether it is positive, square and reciprocal. Also, ensures that the matrix
        does not exceed 15 or 20 rows, depending on the random index. This ensures that
        every Compare object will have a consistency ratio. If all tests pass, it sets
        the 'matrix' and 'shape' properties of the Compare object.
        :param input_matrix: the matrix of the Compare object
        """

        # todo change messages to numbers in a message dictionary
        try:
            matrix = np.matrix(input_matrix)
        except:
            raise AHPException('Input cannot be cast as a matrix')
        shape = matrix.shape[0]

        if (self.random_index == 'saaty' and shape > 15) or shape > 20:
            raise AHPException('Input too large: cannot compute consistency ratio')
        try:
            if (matrix <= 0).any():
                raise AHPException('Input contains values less than one')
        except AttributeError:
            raise AHPException('Input contains invalid values')
        try:
            np.linalg.matrix_power(matrix, 2)
        except ValueError:
            raise AHPException('Input is not square')
        if not (np.multiply(matrix, matrix.T) == np.ones(shape)).all():
            raise AHPException('Input is not reciprocal')

        self.matrix = matrix
        self.shape = shape
        return

    def compute(self):
        try:
            # If the comparison type is quantitative, normalize the input values
            if self.type == 'quant':
                self.normalize()
            # If the comparison type is qualitative, compute both the priority vector and the
            # consistency ratio
            else:
                self.compute_priority_vector(self.matrix, self.iterations)
                self.compute_consistency_ratio()
            # Create the weights dictionary
            comp_dict = dict([(key, val[0]) for key, val in zip(self.criteria, self.priority_vector)])
            self.weights = {self.name: comp_dict}
        except Exception, error:
            raise AHPException(error)

        print 'Compare Name:', self.name
        print 'Consistency:', self.consistency_ratio
        for k, v in self.weights[self.name].iteritems():
            print k, round(v, self.precision)
        print
        return

    def compute_priority_vector(self, matrix, iterations, comp_eigenvector=None):
        """
        Computes the priority vector of a matrix. Sets the 'remainder' and
        'priority_vector' properties of the Compare object.
        :param matrix: numpy matrix, the matrix from which to derive the priority vector
        :param iterations: integer, number of iterations to run before the function stops
        :param comp_eigenvector: numpy array, a comparison eigenvector used during
            recursion; DO NOT MODIFY
        """

        # Compute the principal eigenvector by normalizing the rows of a newly squared matrix
        sq_matrix = np.linalg.matrix_power(matrix, 2)
        row_sum = np.sum(sq_matrix, 1)
        total_sum = np.sum(row_sum)
        princ_eigenvector = np.divide(row_sum, total_sum).round(self.precision)
        # Create a zero matrix as the comparison eigenvector if this is the first iteration
        if comp_eigenvector is None:
            comp_eigenvector = np.zeros(self.shape)
        # Compute the difference between the principal and comparison eigenvectors
        remainder = np.subtract(princ_eigenvector, comp_eigenvector).round(self.precision)
        # If the difference between the two eigenvectors is zero (after rounding to the self.precision variable),
        # set the current principal eigenvector as the priority vector for the matrix and set the difference
        # between the two as the remainder, which will always be a zero matrix
        if not np.any(remainder):
            self.remainder = remainder
            self.priority_vector = princ_eigenvector
            return
        # Recursively run the function until either there is no difference between the principal and
        # comparison eigenvectors, or until the predefined number of iterations has been met, in which
        # case set the last principal eigenvector as the priority vector and set the difference between
        # the two as the remainder, which will be a matrix containing non-zero numbers
        iterations -= 1
        if iterations > 0:
            return self.compute_priority_vector(sq_matrix, iterations, princ_eigenvector)
        else:
            self.remainder = remainder
            self.priority_vector = princ_eigenvector
            return

    def compute_consistency_ratio(self):
        """
        Computes the consistency ratio of the matrix, using random index estimates from
        Donegan and Dodd's 'A note on Saaty's Random Indexes' in Mathematical and Computer
        Modelling, 15:10, 1991, 135-137 (doi: 10.1016/0895-7177(91)90098-R).
        If the random index of the object is set to 'saaty', use the estimates from
        Saaty, Thomas L. 2005. Theory And Applications Of The Analytic Network Process.
        Pittsburgh: RWS Publications, pg. 31.
        Sets the 'consistency_ratio' property of the Compare object.
        """

        # todo how to deal with larger matrices?
        # A valid, square, reciprocal matrix with only one or two rows must be consistent
        if self.shape < 3:
            self.consistency_ratio = 0.0
            return
        # Determine which random index to use
        if self.random_index == 'saaty':
            ri_dict = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45,
                       10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}
        else:
            ri_dict = {3: 0.4914, 4: 0.8286, 5: 1.0591, 6: 1.1797, 7: 1.2519,
                       8: 1.3171, 9: 1.3733, 10: 1.4055, 11: 1.4213, 12: 1.4497,
                       13: 1.4643, 14: 1.4822, 15: 1.4969, 16: 1.5078, 17: 1.5153,
                       18: 1.5262, 19: 1.5313, 20: 1.5371}
        random_index = ri_dict[self.shape]

        try:
            # Find the Perron Frobenius eigenvalue of the matrix
            lambda_max = np.linalg.eigvals(self.matrix).max()
            # Compute the consistency index
            consistency_index = (lambda_max - self.shape) / (self.shape - 1)
            # Compute the consistency ratio
            self.consistency_ratio = (np.real(consistency_index / random_index)).round(self.precision)
            return
        except np.linalg.LinAlgError, error:
            raise AHPException(error)

    def normalize(self):
        """
        Computes the priority vector of a valid matrix by normalizing the input values, then
        sets the consistency ratio to 0.0.
        """

        total_sum = float(np.sum(self.matrix))
        self.priority_vector = np.divide(self.matrix, total_sum).round(self.precision).reshape(len(self.matrix), 1)
        self.consistency_ratio = 0.0
        return


class Compose(object):

    def __init__(self, name=None, parent=None, children=None):
        self.name = name
        self.parent = parent
        self.children = children
        self.weights = dict()
        self.precision = None

        self.compute_precision()
        self.compute_total_priority()
        self.normalize_total_priority()

        # todo remove this
        print 'Compose Name:', self.name
        for k, v in self.weights[self.parent.name].iteritems():
            print k, ':', round(v, self.precision)
        print

    def compute_precision(self):
        """
        Updates the 'precision' property of the Compose object by selecting
        the lowest precision of all its children.
        """

        self.precision = np.min([child.precision for child in self.children])
        return

    def compute_total_priority(self):
        """
        Computes the total priorities of the Compose object's parent criteria
        given the priority vectors of its children. Uses the 'distributive' mode.
        """

        for pk, pv in self.parent.weights[self.parent.name].iteritems():
            for child in self.children:

                if pk in child.weights:
                    for ck, cv in child.weights[pk].iteritems():
                        try:
                            self.weights[ck] += np.multiply(pv, cv)
                        except KeyError:
                            self.weights[ck] = np.multiply(pv, cv)
                    break
        return

    def normalize_total_priority(self):
        """
        Updates the 'weights' property of the Compose object with
        their normalized values.
        """

        total_sum = sum(self.weights.itervalues())
        comp_dict = {key: np.divide(value, total_sum) for key, value in self.weights.iteritems()}
        self.weights = {self.name: comp_dict}
        return


class AHPException(Exception):
    """
    The custom Exception class of the AHP module
    """
    def __init__(self, msg):
        print msg
        exit(1)


if __name__ == '__main__':
    pass

    # Example from Saaty, Thomas, L., Theory and Applications of the Analytic Network Process, 2005
    # crit = np.matrix([[1, .2, 3, .5, 5],
    #                   [5, 1, 7, 1, 7],
    #                   [1/3., 1/7., 1, .25, 3],
    #                   [2, 1, 4, 1, 7],
    #                   [.2, 1/7., 1/3., 1/7., 1]])
    #
    # culture = np.matrix([[1, .5, 1, .5],
    #                      [2, 1, 2.5, 1],
    #                      [1, 1/2.5, 1, 1/2.5],
    #                      [2, 1, 2.5, 1]])
    #
    # family = np.matrix([[1, 2, 1/3., 4],
    #                     [.5, 1, 1/8., 2],
    #                     [3, 8, 1, 9],
    #                     [.25, .5, 1/9., 1]])
    #
    # housing = np.matrix([[1, 5, .5, 2.5],
    #                      [.2, 1, 1/9., .25],
    #                      [2, 9, 1, 7],
    #                      [1/2.5, 4, 1/7., 1]])
    #
    # jobs = np.matrix([[1, .5, 3, 4],
    #                   [2, 1, 6, 8],
    #                   [1/3., 1/6., 1, 1],
    #                   [.25, 1/8., 1, 1]])
    #
    # transportation = np.matrix([[1, 1.5, .5, 4],
    #                             [1/1.5, 1, 1/3.5, 2.5],
    #                             [2, 3.5, 1, 9],
    #                             [.25, 1/2.5, 1/9., 1]])
    #
    # cities = ['Bethesda', 'Boston', 'Pittsburgh', 'Santa Fe']
    # crits = ['Culture', 'Family', 'Housing', 'Jobs', 'Transportation']
    #
    # print 'Saaty'
    # cu = Compare('Culture', culture, cities, 3, random_index='Saaty')
    # f = Compare('Family', family, cities, 3, random_index='Saaty')
    # h = Compare('Housing', housing, cities, 3, random_index='Saaty')
    # j = Compare('Jobs', jobs, cities, 3, random_index='Saaty')
    # t = Compare('Transportation', transportation, cities, 3, random_index='Saaty')
    #
    # comp_matrices = [cu, f, h, j, t]
    # cr = Compare('Criteria', crit, crits, 3, random_index='Saaty')
    # print
    # Compose('Goal', cr, comp_matrices)
    #
    # print '================='
    # print
    # print 'Donegan and Dodd'
    # cu = Compare('Culture', culture, cities, 3)
    # f = Compare('Family', family, cities, 3)
    # h = Compare('Housing', housing, cities, 3)
    # j = Compare('Jobs', jobs, cities, 3)
    # t = Compare('Transportation', transportation, cities, 3)
    #
    # comp_matrices = [cu, f, h, j, t]
    # cr = Compare('Criteria', crit, crits, 3)
    # print
    # Compose('Goal', cr, comp_matrices)

    # Example from https://en.wikipedia.org/wiki/Analytic_hierarchy_process_%E2%80%93_leader_example
    # experience = np.matrix([[1, .25, 4], [4, 1, 9], [.25, 1/9., 1]])
    # education = np.matrix([[1, 3, .2], [1/3., 1, 1/7.], [5, 7, 1]])
    # charisma = np.matrix([[1, 5, 9], [.2, 1, 4], [1/9., .25, 1]])
    # age = np.matrix([[1, 1/3., 5], [3, 1, 9], [.2, 1/9., 1]])
    # criteria = np.matrix([[1, 4, 3, 7], [.25, 1, 1/3., 3], [1/3., 3, 1, 5], [1/7., 1/3., .2, 1]])
    #
    # alt1 = ['Tom', 'Dick', 'Harry']
    #
    # exp = Compare('exp', experience, alt1, 3, random_index='saaty')
    # edu = Compare('edu', education, alt1, 3, random_index='saaty')
    # cha = Compare('cha', charisma, alt1, 3, random_index='saaty')
    # age = Compare('age', age, alt1, 3, random_index='saaty')
    #
    # children = [exp, edu, cha, age]
    #
    # alt2 = ['exp', 'edu', 'cha', 'age']
    #
    # parent = Compare('goal', criteria, alt2, 3, random_index='saaty')
    #
    # Compose('goal', parent, children)

    # Examples from Saaty, Thomas L., 'Decision making with the analytic hierarchy process,'
    # Int. J. Services Sciences, 1:1, 2008, pp. 83-98.
    # drinks_val = np.matrix([[1, 9, 5, 2, 1, 1, .5],
    #                     [1/9., 1, 1/3., 1/9., 1/9., 1/9., 1/9.],
    #                     [.2, 3, 1, 1/3., .25, 1/3., 1/9.],
    #                     [.5, 9, 3, 1, .5, 1, 1/3.],
    #                     [1, 9, 4, 2, 1, 2, .5],
    #                     [1, 9, 3, 1, .5, 1, 1/3.],
    #                     [2, 9, 9, 3, 2, 3, 1]])
    # drinks_cri = ('coffee', 'wine', 'tea', 'beer', 'sodas', 'milk', 'water')
    # Compare('Drinks', drinks_val, drinks_cri, precision=3, random_index='saaty')

    # todo solve this problem
    # salary_m = np.matrix([[1, 4, 3, 6],
    #                       [.25, 1, 3, 5],
    #                       [1/3., 1/3., 1, 2],
    #                       [1/6., .2, .5, 1]])
    # salary_cri = ('Domestic', 'International', 'College', 'University')
    # salary = Compare('salary', salary_m, salary_cri, 3, random_index='Saaty')
    #
    # flex_m = np.matrix([[1, 1/3., 1/6.],
    #                     [3, 1, .25],
    #                     [6, 4, 1]])
    # flex_cri = ('Location', 'Time', 'Work')
    # flexibility = Compare('flexibility', flex_m, flex_cri, 3, random_index='saaty')
    #
    # children = [salary, flexibility]
    #
    # parent_m = np.matrix([[1, .25, 1/6., .25, 1/8.],
    #                       [4, 1, 1/3., 3, 1/7.],
    #                       [6, 3, 1, 4, .5],
    #                       [4, 1/3., .25, 1, 1/7.],
    #                       [8, 7, 2, 7, 1]])
    # parent_cri = ('flexibility', 'opportunities', 'security', 'reputation', 'salary')
    # parent = Compare('Goal', parent_m, parent_cri, 3, random_index='saaty')
    # Compose('Goal', parent, children)

    car_cri = ('civic', 'saturn', 'escort', 'clio')

    # gas_m = np.matrix([[34], [27], [24], [28]])
    # gas = Compare('gas', gas_m, car_cri, 3, comp_type='quant')

    rel_m = np.matrix([[1, 2, 5, 1], [.5, 1, 3, 2], [.2, 1/3., 1, .25], [1, .5, 4, 1]])
    rel = Compare('rel', rel_m, car_cri, 3)

    style_m = np.matrix([[1, .25, 4, 1/6.], [4, 1, 4, .25], [.25, .25, 1, .2], [6, 4, 5, 1]])
    style = Compare('style', style_m, car_cri, 3)

    cri_m = np.matrix([[1, .5, 3], [2, 1, 4], [1/3., .25, 1]])
    cri_cri = ('style', 'rel', 'gas')
    parent = Compare('goal', cri_m, cri_cri)

    # Compose('goal', parent, (style, rel, gas))

    # test_m = '1,2,5,1;.5,1,3,2;.2,1/3,1,.25;1,1/2,4,1'
    #
    # Compare('test', test_m, car_cri)

    def sym(a):
        return a + a.T - np.diag(a.diagonal())

    m2 = '2 5 1; 3 2; 1/4'

    Compare('test', m2, car_cri)
