from __future__ import division
import numpy as np


class Compare(object):
    # todo modify comp_type if need be
    """
    This class computes the priority vector and consistency ratio of a positive
    reciprocal matrix. The 'weights' property contains the priority vector as a dictionary
    whose keys are criteria and whose values are the criteria's priority vectors, or weights.
    The 'CR' property contains the computed consistency ratio of the input matrix as a float.
    The 'remainder' property contains the difference (as a numpy array) between the final two
    eigenvectors used by the compute_priority_vector function.
    :param name: string, the name of the Compare object; if the object has a parent,
        this name MUST exist as a criterion of its parent
    :param matrix: numpy matrix, the matrix from which to derive the priority vector
    :param criteria: list of strings, the criteria of the matrix, in the same left-to-right
        order as the values they name in the input matrix
    :param precision: integer, # of decimal places of precision to compute both the priority
        vector and the consistency ratio; default is 4
    :param comp_type: string, IS THIS NECESSARY???
    :param iters: integer, # of iterations before the compute_eigenvector function stops;
        default is 100
    :param random_index: string, the random index estimates used to compute the consistency ratio;
        valid input: dd, saaty, default is dd
    """

    def __init__(self, name=None, matrix=None, criteria=None,
                 precision=4, comp_type='qual', iters=100, random_index='dd'):
        self.name = name
        self.matrix = np.matrix(matrix)
        self.criteria = criteria
        self.shape = self.matrix.shape[0]
        self.type = comp_type
        self.precision = precision
        self.iterations = iters
        self.RI = random_index.lower()
        self.priority_vector = None
        self.remainder = None
        self.CR = None
        self.weights = None

        self.compute()

    def compute(self):

        if self.matrix is not None and str(self.matrix) != '[[1]]':
            try:
                # If the comparison type is quantitative, normalize the priority vectors
                if self.type == 'quant':
                    self.normalize()
                # If the comparison type is qualitative, compute the priority vector and the
                # consistency ratio
                else:
                    self.compute_priority_vector(self.matrix, self.iterations)
                    self.compute_consistency_ratio()
                # Create the weights dictionary
                comp_dict = dict([(key, val[0]) for key, val in zip(self.criteria, self.priority_vector)])
                self.weights = {self.name: comp_dict}
            except AHPException, err:
                raise AHPException(err)
        else:
            raise AHPException('Input does not contain values for all criteria')

        print 'Compare Name:', self.name
        print 'Consistency:', self.CR
        for k, v in self.weights[self.name].iteritems():
            print k, round(v, self.precision)
        print

    def compute_priority_vector(self, matrix, iterations, comp_eigenvector=None):
        """
        Computes the priority vector of a valid matrix. Sets the 'remainder' and
        'priority_vector' properties of the Compare object.
        :param matrix: numpy matrix, the matrix from which to derive the priority vector
        :param iterations: integer, # of iterations before the function stops
        :param comp_eigenvector: numpy array, a comparison eigenvector
        used during recursion; DO NOT MODIFY
        """

        # todo change messages to numbers in a message dictionary
        try:
            sq_matrix = np.linalg.matrix_power(matrix, 2)
        except ValueError:
            raise AHPException('Input is not square')
        if (self.matrix <= 0).any():
            raise AHPException('Input contains values less than one')
        if not (np.multiply(self.matrix, self.matrix.T) == np.ones(self.shape)).all():
            raise AHPException('Input is not reciprocal')

        with np.errstate(invalid='ignore'):
            row_sum = np.sum(sq_matrix, 1)
            total_sum = np.sum(row_sum)
            princ_eigenvector = np.divide(row_sum, total_sum).round(self.precision)

            if np.isnan(princ_eigenvector).any():
                return
            elif not princ_eigenvector.all():
                shape = princ_eigenvector.shape
                self.remainder = np.zeros(shape)
                self.priority_vector = np.full(shape, np.true_divide(1, shape[0])).round(self.precision)
                return

            if comp_eigenvector is None:
                comp_eigenvector = np.zeros(self.shape)

            remainder = np.subtract(princ_eigenvector, comp_eigenvector).round(self.precision)

        if not np.any(remainder):
            self.remainder = remainder
            self.priority_vector = princ_eigenvector
            return

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
        Sets the 'CR' property of the Compare object.
        """

        if self.shape < 3:
            self.CR = 0.0
            return

        if self.RI == 'saaty':
            ri_dict = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45,
                       10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}
        else:
            ri_dict = {3: 0.4914, 4: 0.8286, 5: 1.0591, 6: 1.1797, 7: 1.2519,
                       8: 1.3171, 9: 1.3733, 10: 1.4055, 11: 1.4213, 12: 1.4497,
                       13: 1.4643, 14: 1.4822, 15: 1.4969, 16: 1.5078, 17: 1.5153,
                       18: 1.5262, 19: 1.5313, 20: 1.5371}
        random_index = ri_dict[self.shape]

        try:
            # Find the Perron–Frobenius eigenvalue of the matrix
            lambda_max = np.linalg.eigvals(self.matrix).max()
            # Compute the consistency index
            consistency_index = (lambda_max - self.shape) / (self.shape - 1)
            # Compute the consistency ratio
            self.CR = (np.real(consistency_index / random_index)).round(self.precision)
        except np.linalg.LinAlgError, error:
            raise AHPException(error)
        finally:
            return

    def normalize(self):
        """
        Computes the priority vector of a valid matrix by normalizing the input values and
        sets the consistency ratio to 0.0.
        """

        total_sum = float(np.sum(self.matrix))
        self.priority_vector = np.divide(self.matrix, total_sum).round(self.precision).reshape(len(self.matrix), 1)
        self.CR = 0.0
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
    # cu = Compare('Culture', cities, culture, 3, ri='Saaty')
    # f = Compare('Family', cities, family, 3, ri='Saaty')
    # h = Compare('Housing', cities, housing, 3, ri='Saaty')
    # j = Compare('Jobs', cities, jobs, 3, ri='Saaty')
    # t = Compare('Transportation', cities, transportation, 3, ri='Saaty')
    #
    # comp_matrices = [cu, f, h, j, t]
    # cr = Compare('Criteria', crits, crit, 3, ri='Saaty')
    # print
    # Compose('Goal', cr, comp_matrices)
    # Compose('Goal', cr, comp_matrices, mode='Ideal')
    #
    # print '================='
    # print
    # print 'Donegan and Dodd'
    # cu = Compare('Culture', cities, culture, 3)
    # f = Compare('Family', cities, family, 3)
    # h = Compare('Housing', cities, housing, 3)
    # j = Compare('Jobs', cities, jobs, 3)
    # t = Compare('Transportation', cities, transportation, 3)
    #
    # comp_matrices = [cu, f, h, j, t]
    # cr = Compare('Criteria', crits, crit, 3)
    # print
    # Compose('Goal', cr, comp_matrices)
    # Compose('Goal', cr, comp_matrices, mode='IDEAL')

    # Example from https://en.wikipedia.org/wiki/Analytic_hierarchy_process_%E2%80%93_leader_example
    experience = np.matrix([[1, .25, 4], [4, 1, 9], [.25, 1/9., 1]])
    education = np.matrix([[1, 3, .2], [1/3., 1, 1/7.], [5, 7, 1]])
    charisma = np.matrix([[1, 5, 9], [.2, 1, 4], [1/9., .25, 1]])
    age = np.matrix([[1, 1/3., 5], [3, 1, 9], [.2, 1/9., 1]])
    criteria = np.matrix([[1, 4, 3, 7], [.25, 1, 1/3., 3], [1/3., 3, 1, 5], [1/7., 1/3., .2, 1]])

    alt1 = ['Tom', 'Dick', 'Harry']

    exp = Compare('exp', experience, alt1, 3, random_index='saaty')
    edu = Compare('edu', education, alt1, 3, random_index='saaty')
    cha = Compare('cha', charisma, alt1, 3, random_index='saaty')
    age = Compare('age', age, alt1, 3, random_index='saaty')

    children = [exp, edu, cha, age]

    alt2 = ['exp', 'edu', 'cha', 'age']

    parent = Compare('goal', criteria, alt2, 3, random_index='saaty')

    Compose('goal', parent, children)

    def convert(matrix):
        new_matrix = []
        try:
            for x in matrix.replace(',', ' ').split(';'):
                new_matrix.append([eval(y, {'__builtin__': None}, {}) for y in x.split()])
        except ValueError:
            raise AHPException('The input matrix should follow the numpy form "1 2; 3 4" or "1, 2; 3, 4"')
        return new_matrix





