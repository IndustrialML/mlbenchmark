
class DataProvider(object):

    def __init__(self, x, y):
        self.x = x
        self.y_true = y

    def __iter__(self):
        self.idx = 0
        return self


    def __next__(self):
        if self.idx >= len(self.x):
            raise StopIteration

        x = self.x[self.idx]
        y = self.y_true[self.idx]
        self.idx += 1
        
        return x, y