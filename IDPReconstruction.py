#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 23:16:59 2022

@author: prottayprotivash
"""

import numpy as np
import pandas as pd
import copy
from ProtectedDataset import ProtectedDataset


class LocalSensitivityReconstructionAttacker:
    """ LocalSensitivityReconstructionAttacker runs experiments to reconstruct a protected dataset based on results of
        the binary count query. The constructor takes:
        :param protectedDataset: ProtectedDataset that answers count queries using individual differential privacy
        :param precision: Float that represents the level of precision each element will be reconstructed to
        :param bounds: Dictionary containing {key=string column name : value=(float lower bound, float upper bound) }
        :param reconstructionOrder: List containing [string of column name] representing the order by which to reconstruct columns
    """
    def __init__(self, protectedDataset, precision, bounds, reconstructionOrder):
        # set up access to and known information about protected dataset
        self.queryAnswerer = protectedDataset
        self.categoricalColumns = queryAnswerer.categoricalColumns
        self.numericalColumns = queryAnswerer.numericalColumns
        self.reconstructionOrderColumns = reconstructionOrder
        self.dataOrderColumns = self.queryAnswerer.getColumns()
        self.epsilon = 1

        # set up reconstruction parameters
        self.precision = precision
        self.bounds = bounds
        self.Reconstructed_df = {}

        # set up other helpful tracking information
        self.Reconstructed_column_df_dict = {}
        self.column_only_uniques_dictionary = {}
        self.total_uniques_dictionary = {}
        self.column_only_query_dictionary = {}
        self.total_query_dictionary = {}

    @staticmethod
    def putElementsIntoDatabase(element, database, count):
        """
        Helper method to insert a given number of elements into a list.
        :param element: The element to insert into the database
        :param database: List to insert the given element into
        :param count: Number of times to insert the given element into the list
        """
        for i in range(0, count):
            database.append(element)

    @staticmethod
    def listOfLists(lst):
        """
        Helper method to create a list of lists out of the elements from the given list.
        :param lst: List to create a list of lists out of
        """
        return list(map(lambda el: [el], lst))

    def binarySearchRightCountBoundary(self, U, V, low, high, colNumber, comparisonDict):
        """
        Binary search to find the upper side of the boundary between decimal and binary responses to the count query
            with varied values of b.
        :param U: float to pass as u to the count query
        :param V: float to pass as v to the count query
        :param low: int lower boundary of the binary search
        :param high: int upper boundary of the binary search
        :param colNumber: int column number being reconstructed
        :param comparisonDict: Dictionary containing {key=string column name : value=(float u, float v) }
            for conditioning on previous columns
        :return: int count that represents the upper side of the decimal-binary boundary
        """
        # Check base case
        if high >= low:
            mid = (high + low) // 2

            x = self.queryAnswerer.count(U, V, mid, self.epsilon, colNumber, comparisonDict)
            if x == 0:  # x is 0, so count + k >= mid, so we must shrink downwards
                y = self.queryAnswerer.count(U, V, mid - 1, self.epsilon, colNumber, comparisonDict)
                if y != 0:
                    return mid
                else:
                    return self.binarySearchRightCountBoundary(U, V, low, mid - 1, colNumber, comparisonDict)
            elif x != 1:  # x is decimal
                # observing a decimal value bounds the search space for the true boundary
                return self.binarySearchRightCountBoundary(U, V, mid + 1, min(high, mid + self.queryAnswerer.k + 1), colNumber, comparisonDict)
            else:  # x is 1, so count - k < mid, so we must shrink upwards
                return self.binarySearchRightCountBoundary(U, V, mid + 1, high, colNumber, comparisonDict)
        else:
            return -1

    def reconstructCount(self, U, V, countLimit, colNumber, comparisonDict):
        """
        Reconstructs the number of elements in a given range
        :param U: float u in the count query
        :param V: float v in the count query
        :param countLimit: int known initialize upper limit of the count
        :param colNumber: int column number being reconstructed
        :param comparisonDict: Dictionary containing {key=string column name : value=(float u, float v) }
            for conditioning on previous columns
        :return: int the count of elements in the range [U, V)
        """
        RBoundaryCount = self.binarySearchRightCountBoundary(U, V, 0, countLimit, colNumber, comparisonDict)
        return RBoundaryCount - self.queryAnswerer.k

    def binarySearchValueReconstruction(self, U, V, low, high, currCount, colNumber, comparisonDict):
        """
        Finds the next largest value in the given column and the count of remaining elements to reconstruct
        :param U: float u to be passed to the count query
        :param V: float v unused
        :param low: float the lower boundary of the binary search
        :param high: float the upper boundary of the binary search
        :param currCount: int the remaining number of elements in the column to be reconstructed
        :param colNumber: int column number being reconstructed
        :param comparisonDict: Dictionary containing {key=string column name : value=(float u, float v) }
            for conditioning on previous columns
        :return: (value, count) representing the next largest value to reconstruct and the count of elements in the range up to, but excluding count
        """
        if high >= low:
            mid = (high + low) // 2

            if currCount + self.queryAnswerer.k - 1 < self.queryAnswerer.numRows:  # we assume we know how many rows are in the database, but this could have been previously reconstructed even if not known originally
                x = self.queryAnswerer.count(U, mid, currCount + self.queryAnswerer.k - 1, self.epsilon, colNumber, comparisonDict)
                if x == 0:  # count_[U, mid) <= currCount - 1 < currCount, so value must shrink upwards
                    return self.binarySearchValueReconstruction(U, V, mid + self.precision, high, currCount, colNumber, comparisonDict)
                else:  # since count_[U, mid) cannot be greater than currCount, it must be equal to currCount
                    y = self.queryAnswerer.count(U, mid - self.precision, currCount + self.queryAnswerer.k - 1, self.epsilon, colNumber, comparisonDict)
                    if y == 0:  # count_[U, mid-precision) <= currCount - 1 < currCount
                        # since the count in wider range is equal to currCount, and count in lower range is not, we have found the next largest value
                        elementCount = self.reconstructCount(U, mid - self.precision, currCount, colNumber, comparisonDict)
                        return mid - self.precision, elementCount
                    else:  # count_[U, mid-precision) >= currCount, so we have not found a boundary, and must shrink downwards
                        return self.binarySearchValueReconstruction(U, V, low, mid, currCount, colNumber, comparisonDict)

            # normally, we search for an upper boundary, but need to adapt when currCount is too high for that boundary to occur
            else:
                # shorter range query, which should be 1 when its count is DSize, then when it's decimal or 0, we check the larger range.
                y = self.queryAnswerer.count(U, mid - self.precision, currCount - self.queryAnswerer.k - 1, self.epsilon, colNumber, comparisonDict)
                if y == 1:  # count_[U, mid-precision) must be numRows, so we need to shrink downwards
                    return self.binarySearchValueReconstruction(U, V, low, mid - self.precision, currCount, colNumber, comparisonDict)
                else:  # count_[U, mid-precision) < numRows because it cannot be greater than numRows
                    x = self.queryAnswerer.count(U, mid, currCount - self.queryAnswerer.k - 1, self.epsilon, colNumber, comparisonDict)
                    if x == 1:  # count_[U, mid) must be numRows, which means we have reconstructed the value
                        elementCount = self.reconstructCount(U, mid - self.precision, currCount, colNumber, comparisonDict)
                        return mid - self.precision, elementCount
                    else:  # count_[U, mid) < numRows, so the element found is not the next biggest, so we must shrink upwards
                        return self.binarySearchValueReconstruction(U, V, mid + self.precision, high, currCount, colNumber, comparisonDict)
        else:
            return -1, 0

    def binarySearchValueReconstructionSingleElement(self, U, low, high, colNumber, comparisonDict):
        """
        Reconstructs the element in the column while assuming that there is only one such element that fits the preconditions
            (used specifically for the targeting experiment)
        :param U: float u to pass to the count query
        :param low: float the lower bound of the binary search
        :param high: float the upper bound of the binary search
        :param colNumber: int the column number being reconstructed
        :param comparisonDict: Dictionary containing {key=string column name : value=(float u, float v) }
            for conditioning on previous columns
        :return: the element fitting the preconditions in the given column
        """
        if high >= low:
            mid = (high + low) // 2

            x = self.queryAnswerer.count(U, mid, self.queryAnswerer.k, self.epsilon, colNumber, comparisonDict)
            if x == 0:  # currCount-1 >= count from U to mid >= count from U to mid-precision, so we must shrink upwards
                return self.binarySearchValueReconstructionSingleElement(U, mid + self.precision, high, colNumber, comparisonDict)
            else:
                y = self.queryAnswerer.count(U, mid - self.precision, self.queryAnswerer.k, self.epsilon, colNumber, comparisonDict)
                if y == 0:  # currCount-1 < count from U to mid-precision, <= count from U to mid, so we must shrink downwards
                    return mid - self.precision
                else:
                    return self.binarySearchValueReconstructionSingleElement(U, low, mid, colNumber, comparisonDict)
        else:
            return -1

    def reconstructValueSet(self, minn, maxx, elementCount, colNumber, comparisonList):
        """
        Reconstructs all the values in the given column that fit the preconditions in the given comparisonList
        :param minn: float the lower boundary for the values in the given column
        :param maxx: float the upper boundary for the values in the given column
        :param elementCount: int the number of elements known to fit the preconditions
        :param colNumber: int the column number being reconstructed
        :param comparisonDict: Dictionary containing {key=string column name : value=(float u, float v) }
            for conditioning on previous columns
        :return: Reconstructed dataset for the given column and preconditions
        """
        U, V = minn - 1, maxx + 1
        ReconstructedDatabase = []
        currCount = elementCount
        prevCount = currCount
        while currCount >= 1:
            element, currCount = self.binarySearchValueReconstruction(U, V, U, V, currCount, colNumber, comparisonList)
            V = element
            count = prevCount - currCount
            self.putElementsIntoDatabase(element, ReconstructedDatabase, count)
            prevCount = currCount
        return ReconstructedDatabase

    def individualColumnReconstructionExperiment(self):
        """
        Conducts an experiment for reconstructing each individual column
        :return: None
        """
        self.queryAnswerer.resetQueryCounter()
        QueriesByColumn = 0
        maxx = np.Infinity
        minn = np.Infinity
        ReconDatabase = []
        for i in range(len(self.reconstructionOrderColumns)):
            name = self.reconstructionOrderColumns[i]
            if self.reconstructionOrderColumns[i] in self.bounds.keys():
                minn = self.bounds[self.reconstructionOrderColumns[i]][0]
                maxx = self.bounds[self.reconstructionOrderColumns[i]][1]
            else:
                categoricalBounds = self.queryAnswerer.getCategoricalBounds(self.reconstructionOrderColumns[i])
                minn = categoricalBounds[0]
                maxx = categoricalBounds[1]
            ReconDatabase.append(self.reconstructValueSet(minn, maxx, self.queryAnswerer.numRows, self.dataOrderColumns.index(self.reconstructionOrderColumns[i]), None))

            for a in ReconDatabase:
                self.Reconstructed_column_df_dict[self.reconstructionOrderColumns[i]] = a
                ReconDatabase = copy.deepcopy(self.listOfLists(a))

            df = pd.DataFrame(ReconDatabase)
            new_set = set(self.Reconstructed_column_df_dict[self.reconstructionOrderColumns[i]])
            print('######################## Reconstructed Column ############################')
            print('Column Name:',self.reconstructionOrderColumns[i])
            print(df)
            print('-------------Query Statistics-------------')
            print('Total Queries Used So Far:', self.queryAnswerer.getNumQueries())
            print('Total Queries Needed for column:', self.queryAnswerer.getNumQueries() - QueriesByColumn)
            print('Number of Unique Elements in Column:', len(new_set))
            print('##########################################################################')
            print('\n')

            self.column_only_query_dictionary[self.reconstructionOrderColumns[i]] = self.queryAnswerer.getNumQueries() - QueriesByColumn
            self.column_only_uniques_dictionary[self.reconstructionOrderColumns[i]] = len(new_set)

            QueriesByColumn = self.queryAnswerer.getNumQueries()

        print('######################## Column Reconstructed Dataframe ############################')
        print(pd.DataFrame(self.Reconstructed_column_df_dict))
        print('-------------Query Statistics-------------')
        print('Total Queries Used For Reconstruction:', sum(self.column_only_query_dictionary.values()))
        print('Total Unique Elements Reoconstructed:', sum(self.column_only_uniques_dictionary.values()))
        print('##########################################################################\n')

    def entireDatabaseReconstructionExperiment(self):
        """
        Conducts an experiment to reconstruct the entire database
        :return: None
        """
        self.queryAnswerer.resetQueryCounter()
        minn, maxx = np.Infinity, np.Infinity
        QueriesByColumn = 0
        ReconDatabase = []
        CorrectedSequenceList = []
        columnSet = list(map(list, set(map(tuple, ReconDatabase))))

        for i in range(len(self.reconstructionOrderColumns)):
            # set bounds for values, based on either reasonable numerical bounds or the categorical schema
            if self.reconstructionOrderColumns[i] in self.bounds.keys():
                minn = self.bounds[self.reconstructionOrderColumns[i]][0]
                maxx = self.bounds[self.reconstructionOrderColumns[i]][1]  # 1000000
            else:
                categoricalBounds = self.queryAnswerer.getCategoricalBounds(self.reconstructionOrderColumns[i])
                minn = categoricalBounds[0]
                maxx = categoricalBounds[1]

            # if first column, where previous comparisons haven't been constructed yet
            if i == 0:
                ReconDatabase.append(self.reconstructValueSet(minn, maxx, self.queryAnswerer.numRows, self.dataOrderColumns.index(self.reconstructionOrderColumns[i]), None))
                for a in ReconDatabase:
                    ReconDatabase = copy.deepcopy(self.listOfLists(a))
                CorrectedSequenceList = ReconDatabase.copy()
                columnSet = list(map(list, set(map(tuple, ReconDatabase))))
            else:
                CorrectedSequenceList = []
                columnSet = list(map(list, set(map(tuple, ReconDatabase))))

                for j, l1 in enumerate(columnSet):
                    countOfElements = ReconDatabase.count(l1[0:i])

                    # construct comparisons dictionary of column number to range values
                    comparisonDict = {}
                    for col in range(i):
                        comparisonDict[self.queryAnswerer.getColumns().index(self.reconstructionOrderColumns[col])] = (l1[col], l1[col] + self.precision)
                    tempplist = self.reconstructValueSet(minn, maxx, countOfElements, self.dataOrderColumns.index(self.reconstructionOrderColumns[i]), comparisonDict)

                    for a in tempplist:
                        t = l1[:i]
                        t.append(a)
                        CorrectedSequenceList.append(t)

                ReconDatabase = []
                ReconDatabase = copy.deepcopy(CorrectedSequenceList)
                columnSet = list(map(list, set(map(tuple, ReconDatabase))))

            # print analytics
            self.Reconstructed_df = pd.DataFrame(CorrectedSequenceList)
            print('######################## Reconstructed Database So Far ############################')
            print(self.Reconstructed_df)
            print('-------------Query Statistics-------------')
            print('Total Queries Used So Far:', self.queryAnswerer.getNumQueries())
            print('Total Queries Needed for Current column:', self.queryAnswerer.getNumQueries() - QueriesByColumn)
            self.total_query_dictionary[self.reconstructionOrderColumns[i]] = self.queryAnswerer.getNumQueries() - QueriesByColumn
            print('Number of Unique Sequences:', len(columnSet))
            print('##########################################################################')
            print('\n')
            self.total_uniques_dictionary[self.reconstructionOrderColumns[i]] = len(columnSet)
            QueriesByColumn = self.queryAnswerer.getNumQueries()
        self.Reconstructed_df.columns = self.reconstructionOrderColumns
        print('######################## Reconstructed Dataframe ############################')
        print(self.Reconstructed_df)
        print('-------------Query Statistics-------------')
        print('Total Queries Used For Reconstruction:', sum(self.total_query_dictionary.values()))
        print('Total Unique Sequences Reoconstructed:', sum(self.total_uniques_dictionary.values()))
        print('##########################################################################\n')

    def targetingExperiment(self):
        """
        Conducts an experiment that targets users known to be a unique combinations of specified linking columns
        :return: None
        """
        linkingColumns = ["age", "marital", "education", "job", "housing"]
        cols = self.reconstructionOrderColumns.copy()
        for col in linkingColumns:
            cols.remove(col)

        # Finding unique combinations to reconstruct requires accessing the dataset directly to set up the experiment.
        counter = {}
        for i in range(self.queryAnswerer._cat_df.shape[0]):
            key = "|".join([str(self.queryAnswerer._cat_df.at[i, col]) for col in linkingColumns])
            counter[key] = counter.get(key, 0) + 1
        uniques = [x for x in counter if counter[x] == 1]
        # print(uniques)

        # run a reconstruction experiment on each combination
        totalQueryCounts = []
        balanceQueryCounts = []
        for element in uniques:
            self.queryAnswerer.resetQueryCounter()
            QueriesByColumn = 0
            targetSet = [int(x) for x in element.split("|")]
            # construct comparisons dictionary of column number to range values
            comparisonDict = {}
            for i in range(len(linkingColumns)):
                comparisonDict[self.dataOrderColumns.index(linkingColumns[i])] = (targetSet[i], targetSet[i] + self.precision)

            # reconstruct the value in each column in the given row
            for column in range(len(cols)):
                name = cols[column]

                # set bounds for values, based on either reasonable numerical bounds or the categorical schema
                if cols[column] in self.bounds.keys():
                    minn = self.bounds[cols[column]][0]
                    maxx = self.bounds[cols[column]][1]  # 1000000
                else:
                    bounds = self.queryAnswerer.getCategoricalBounds(cols[column])
                    minn = bounds[0]
                    maxx = bounds[1]

                # call the reconstruction
                element = self.binarySearchValueReconstructionSingleElement(-10000000000, minn - 1, maxx + 1, self.dataOrderColumns.index(cols[column]), comparisonDict)
                targetSet.append(element)

                # print(targetSet)
                # print('Total Queries So Far:', self.queryAnswerer.getNumQueries())
                # print('Total Queries for column {0}: {1}'.format(cols[column], self.queryAnswerer.getNumQueries() - QueriesByColumn))
                # print('\n')
                if cols[column] == "balance":
                    balanceQueryCounts.append(self.queryAnswerer.getNumQueries() - QueriesByColumn)
                QueriesByColumn = self.queryAnswerer.getNumQueries()
            totalQueryCounts.append(self.queryAnswerer.getNumQueries())
            # print(targetSet)
            # print('{0} queries for rest of record, {1} queries for balance column\n'.format(totalQueryCounts[-1], balanceQueryCounts[-1]))
        print('\n')
        print('{0} {1} {2}'.format(len(uniques), len(totalQueryCounts), len(balanceQueryCounts)))
        print('Average queries for entire row:', np.array(totalQueryCounts).sum() / len(totalQueryCounts))
        print('Average queries for balance column:', np.array(balanceQueryCounts).sum() / len(balanceQueryCounts))


if __name__ == "__main__":
    categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    numericalColumns = ['day', 'campaign', 'pdays', 'previous', 'age', 'duration', 'balance']
    queryAnswerer = ProtectedDataset('bank-full.csv', 1, categoricalColumns, numericalColumns)

    # setup initial boundaries for attacking numerical columns
    bounds = {'day': (0, 31), 'campaign': (0, 100), 'pdays': (-1, 2000), 'previous': (0, 2000),
              'age': (0, 125), 'duration': (0, 10 ** 4), 'balance': (-10 ** 5, 10 ** 6)}
    reconstructionOrder = ['y','default','housing','loan','marital','contact','poutcome','education','job','month','day','previous','campaign','age','pdays','duration','balance']
    # reconstructionOrder = ['age', 'y', 'default', 'housing', 'loan', 'marital', 'contact', 'poutcome', 'education', 'job',
    #                        'month', 'day', 'previous', 'campaign', 'pdays', 'duration', 'balance']
    attacker = LocalSensitivityReconstructionAttacker(queryAnswerer, 1, bounds, reconstructionOrder)
    attacker.individualColumnReconstructionExperiment()
    attacker.entireDatabaseReconstructionExperiment()
    attacker.targetingExperiment()
    print('--------------------------------------------')
    print('Column Reconstruction')
    print('--------------------------------------------')
    print('\nUnique Elements:')
    print(attacker.column_only_uniques_dictionary)
    print('\nQueries Needed:')
    print(attacker.column_only_query_dictionary)
    print('\nTotal Queries Needed:')
    print(sum(attacker.column_only_query_dictionary.values()))
    print('--------------------------------------------')
    print('Entire Database Reconstruction')
    print('--------------------------------------------')
    print('\nUnique Sequences:')
    print(attacker.total_uniques_dictionary)
    print('\nQueries Needed:')
    print(attacker.total_query_dictionary)
    print('\nTotal Queries Needed:')
    print(sum(attacker.total_query_dictionary.values()))
    print('--------------------------------------------')