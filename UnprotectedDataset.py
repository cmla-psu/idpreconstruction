import numpy as np
import pandas as pd
import functools
import copy


class UnprotectedDataset:
    """
    ProtectedDataset answers count queries on the given dataset honestly. The constructor takes:
    :param file_name: string filename of the dataset to answer queries on
    :param categoricalColumns: List containing [string column name] of categorical columns that must be converted to a numerical representation
    :param numericalColumns: List containing [string column name] of numerical columns
    """

    def __init__(self, file_name, categoricalColumns, numericalColumns):
        self.queriesAnswered = 0
        self.__df = pd.read_csv(file_name, sep=';')

        self.categoricalColumns = categoricalColumns
        self.numericalColumns = numericalColumns
        self._cat_df = self.convertCategoricalToNumbering()  # is protected instead of private because direct access is needed to set up targeting experiment
        self.__cat_df_list = np.array(self._cat_df).T.tolist()
        self.__cat_df_list_T = np.array(self._cat_df).tolist()
        self.columns = self._cat_df.columns.tolist()
        self.numRows = self.__df.shape[0]
        self.numColumns = self.__df.shape[1]

        # setup to make the processing of the count queries faster
        self.__columnSubset = []
        self.__tempDataSubset = []
        self.__colIndex = -1
        self.__storedCount = -1
        self.__comparisonListV2 = [np.Infinity]
        self.__previousComparisonDict = {}

        # print debugging
        # print('-----------------Original dataframe----------------')
        # print(self.__df)
        # print(self.__cat_df)

    # models the data custodian's answer to the given count query
    def count(self, u, v, b, column, comparisonDict=None):
        """
        Models the data custodian's response to the count query
        :param u: float u
        :param v: float v
        :param b: int b
        :param column: int main column number being answered on
        :param comparisonDict: Dictionary containing {key=string column name : value=(float u, float v) }
            for conditioning on previous columns
        :return:
        """
        # u, v, and column are separate from the comparisons purely for purposes of optimizing the speed of the answering based on the experiment
        # they would likely be lumped into the comparisons dictionary in a real thing

        self.queriesAnswered += 1
        count = 0

        # comparisons is a dictionary of column numbers to U and V values

        # construct the subset of the dataset that fits the query bounds only if it is different from the previous query
        # (otherwise, used already constructed subset)
        # This check and storage of the database subset is quality-of-life to speed up the experiment's runtime,
        # but likely would not exist in a real query-answering mechanism
        self.__colIndex = column

        if comparisonDict is None:  # single column, can ignore the comparisons to previous columns
            self.__previousComparisonDict = None
            for element in self.__cat_df_list[self.__colIndex]:
                if u <= element < v:
                    count += 1
        else:  # multiple columns are included in the query
            if not self.__previousComparisonDict == comparisonDict:  # check exists purely for experiment speed purposes
                self.__tempDataSubset = []
                self.__previousComparisonDict = copy.deepcopy(comparisonDict)
                for row in self.__cat_df_list_T:
                    # check if the values in the given columns are within the prescribed boundaries
                    if functools.reduce(lambda x, y: x and y, map(lambda z: z[1][0] <= row[z[0]] < z[1][1], comparisonDict.items()), True):
                    # for item in comparisonDict.items():
                    #     if item[1][0] <= row[item[0]] < item[1][1]:
                        self.__tempDataSubset.append(row)
            # compute the number of elements in the comparison subset that match the parameters for the "main" column
            for row in self.__tempDataSubset:
                if u <= row[column] < v:
                    count += 1

        if count > b:
            return 1
        else:
            return 0

    def convertCategoricalToNumbering(self):
        """
        Converts categorical values to a numerical schema representation
        :return: DataFrame with numerical representation of categorical values
        """
        cat_df = pd.DataFrame()
        for column in self.categoricalColumns:
            cat_df[column] = pd.Categorical(self.__df[column]).codes
        for column in self.numericalColumns:
            cat_df[column] = self.__df[column]
        return cat_df

    def getNumQueries(self):
        """
        :return: number of queries that have been answered so far
        """
        return self.queriesAnswered

    def resetQueryCounter(self):
        """
        Resets the counter for number of queries answered
            (for use between experiments)
        :return: None
        """
        self.queriesAnswered = 0

    def getColumns(self):
        """
        :return: List containing [string column name] column names of the protected dataset
        """
        return self._cat_df.columns.tolist()

    # return the maximum and minimum in the categorical schema for the given column
    def getCategoricalBounds(self, column):
        """
        Gives the bounds for the numerical schema for the given categorical column
        :param column: string column name
        :return: Tuple containing (int lower bound, int upper bound)
        """
        if column in self.categoricalColumns:
            index = self.columns.index(column)
            return min(self.__cat_df_list[index]), max(self.__cat_df_list[index])
        raise NotImplementedError
