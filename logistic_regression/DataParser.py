#!/usr/bin/env python
# encoding=utf-8
import numpy as np


class DataParser:
    """
    Data parser for common formated data file. For example: header=True and separator = ',' means csv file
    Output encode: utf-8
    """

    def __init__(self, filepath, sep='\t', header=False):
        self.filepath = filepath
        self.file = open(filepath, 'rb')
        self.sep = sep
        self.header = header
        self.column_names = range(0, 100)

    def __del__(self):
        self.file.close()

    def readrows(self):
        while True:
            line = self.file.readline()
            if not line:
                break
            # 不能用unicode，因为涉及到中文urlencode和urldecode
            line = line.replace('\n', '')
            arr = line.split(self.sep)
            if self.header:
                self.header = False
                self.column_names = arr
                continue
            row = {}
            for i in range(0, len(arr)):
                row[self.column_names[i]] = arr[i]
            yield row

    def load_data(self):
        """
        Read data and return numpy.array
        """
        data = []
        for row in self.readrows():
            tmp = []
            for key in self.column_names:
                if key == 'density' or key == 'sugar_content':
                    tmp.append(float(row[key]))
                else:
                    tmp.append(int(row[key]))
            data.append(tmp)
        return np.array(data)


