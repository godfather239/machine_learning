#!/usr/bin/env python
# encoding=utf-8


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



