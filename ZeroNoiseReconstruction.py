#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import functools
import copy
from UnprotectedDataset import UnprotectedDataset

class ZeroNoiseReconstructionAttacker:

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

    def laplaceMechanism(self, x):
        x += np.random.laplace(0, 1.0 / self.epsilon, 1)[0]
        return x

    @staticmethod
    def putElementsIntoDatabase(Element, Database, Count):
        for i in range(0, Count):
            Database.append(Element)

    @staticmethod
    def listOfLists(lst):
        return list(map(lambda el: [el], lst))

    def binarySearchRightCountBoundary(self, U, V, low, high, colNumber, comparisonDict):
        # Check base case
        if high >= low:

            mid = (high + low) // 2

            x = self.queryAnswerer.count(U, V, mid, colNumber, comparisonDict)
            if x == 1:
                return self.binarySearchRightCountBoundary(U, V, mid + 1, high, colNumber, comparisonDict)
            else:
                y = self.queryAnswerer.count(U, V, mid - 1, colNumber, comparisonDict)
                if y == 0:
                    return self.binarySearchRightCountBoundary(U, V, low, mid - 1, colNumber, comparisonDict)
                else:
                    # x = 0 and y = 1
                    return mid
        else:
            return -2

    def reconstructCount(self, U, V, countLimit, colNumber, comparisonDict):
        return self.binarySearchRightCountBoundary(U, V, 0, countLimit, colNumber, comparisonDict)

    def binarySearchValueReconstruction(self, U, V, low, high, currCount, colNumber, comparisonDict):
        if high >= low:
            mid = (high + low) // 2

            x = self.queryAnswerer.count(U, mid, currCount - 1, colNumber, comparisonDict)
            # Count2D(Database, U, mid, currCount-1, DSize, comparisonList)
            if x == 0:  # currCount-1 >= count from U to mid >= count from U to mid-increment, so we must shrink upwards
                return self.binarySearchValueReconstruction(U, V, mid + self.precision, high, currCount, colNumber, comparisonDict)
            else:
                y = self.queryAnswerer.count(U, mid - self.precision, currCount - 1, colNumber, comparisonDict)
                if y == 1:  # currCount-1 < count from U to mid-increment, <= count from U to mid, so we must shrink downwards
                    return self.binarySearchValueReconstruction(U, V, low, mid, currCount, colNumber, comparisonDict)
                else:
                    elementCount = self.reconstructCount(U, mid - self.precision, currCount, colNumber, comparisonDict)
                    return mid - self.precision, elementCount
        else:
            return -1, 0

    def reconstructValueSet(self, minn, maxx, elementCount, colNumber, comparisonDict):
        U, V = minn - 1, maxx + 1
        ReconstructedDatabase = []
        currCount = elementCount
        prevCount = currCount
        while currCount >= 1:
            element, currCount = self.binarySearchValueReconstruction(U, V, U, V, currCount, colNumber, comparisonDict)
            V = element
            Count = prevCount - currCount
            self.putElementsIntoDatabase(element, ReconstructedDatabase, Count)
            prevCount = currCount

        return ReconstructedDatabase

    def binarySearchValueReconstructionSingleElement(self, U, low, high, colNumber, comparisonDict):
        if high >= low:
            mid = (high + low) // 2

            x = self.queryAnswerer.count(U, mid, 0, colNumber, comparisonDict)
            # Count2D(Database, U, mid, currCount-1, DSize, comparisonList)
            if x == 0:  # currCount-1 >= count from U to mid >= count from U to mid-increment, so we must shrink upwards
                return self.binarySearchValueReconstructionSingleElement(U, mid + self.precision, high, colNumber, comparisonDict)
            else:
                y = self.queryAnswerer.count(U, mid - self.precision, 0, colNumber, comparisonDict)
                if y == 1:  # currCount-1 < count from U to mid-increment, <= count from U to mid, so we must shrink downwards
                    return self.binarySearchValueReconstructionSingleElement(U, low, mid, colNumber, comparisonDict)
                else:
                    return mid - self.precision
        else:
            return 0

    def individualColumnReconstructionExperiment(self):
        self.queryAnswerer.resetQueryCounter()
        QueriesByColumn = 0
        maxx = np.Infinity
        minn = np.Infinity
        ReconDatabase = []
        for i in range(len(self.reconstructionOrderColumns)):
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

                    # do the reconstruction
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

                # print stats
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
    queryAnswerer = UnprotectedDataset('bank-full.csv', 1, categoricalColumns, numericalColumns)

    # setup initial boundaries for attacking numerical columns
    bounds = {'day': (0, 31), 'campaign': (0, 100), 'pdays': (-1, 2000), 'previous': (0, 2000),
              'age': (0, 125), 'duration': (0, 10 ** 4), 'balance': (-10 ** 5, 10 ** 6)}
    reconstructionOrder = ['y','default','housing','loan','marital','contact','poutcome','education','job','month','day','previous','campaign','age','pdays','duration','balance']
    attacker = ZeroNoiseReconstructionAttacker(queryAnswerer, 1, bounds, reconstructionOrder)
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