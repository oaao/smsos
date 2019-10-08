import os
import pandas

class CsvDataInput:
    """
    Ingests a classified CSV dataset as a pandas dataframe
    and distributes items into training and test data
    according to a given item count or decimal proportion.
    """

    def __init__(self, filepath, training_cases=None, encoding='utf-8'):

        self.data = self._load_csv(filepath, encoding)
        self.split_index = self._get_data_split(training_cases)

        self.train, self.test = self.split_data()

    def split_data(self):
        return (self.data[self.split_index:], self.data[:self.split_index])

    def _load_csv(self, path, encoding):

        realpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

        data = pandas.read_csv(realpath, encoding=encoding)

        try:
            if data.count == 0:
                raise ValueError('Dataframe is empty')
            else:
                return data
        except AttributeError:
            raise AttributeError('Data coult not be loaded as a dataframe')

    def _get_data_split(self, n):

        rows, cols = self.data.shape

        if isinstance(n, int):
            if n <= rows:
                return n
            else:
                raise ValueError('Training case count cannot exceed total data length')

        elif isinstance(n, float):
            if n <= 1.0:
                # casting float to int is a floor operation
                return int(n * rows)
            else:
                raise ValueError('Training case proportion cannot exceed 100%')
        else:
            raise ValueError('Unsupported value; please use count integer or float proportion')
