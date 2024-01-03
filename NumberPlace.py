import numpy as np

class NumberPlace:
    def __init__(self):
        self.number_table = np.zeros((9, 9), dtype=np.int32)

    def set(self, i, j, value): 
        self.number_table[i][j] = value
    
    def get(self): 
        return self.number_table, self.input_table
    
    def check_all(self):
        self.input_table = np.copy(self.number_table) 
        print(self.number_table) 
        result = self.check(0)
        return result

    def check(self, n):
        if n >= 9 * 9:
            return True
        i = n // 9
        j = n % 9
        l = self.number_table[i][j]
        #print(i, j, l)

        if l != 0:
            self.number_table[i][j] = 0
            if self.check3(i, j, l):
                self.number_table[i][j] = l
                return self.check(n + 1)
            self.number_table[i][j] = l
            return False
        for k in range(1,10):
            #print(i, j, k)
            if self.check3(i, j, k):
                self.number_table[i][j] = k
                if self.check(n + 1):
                    return True
                else:
                    self.number_table[i][j] = 0
        return False

    def check3(self, i, j, k):
        if self.check_box(i, j, k):
            if self.check_row(i, j, k):
                if self.check_column(i, j, k):
                    return True
                else:
                    return False
        return False

    def check_box(self, i, j, k): 
        box_row = i // 3
        box_column = j // 3
        rs = box_row * 3
        cs = box_column * 3
        box_list = self.number_table[rs:rs + 3, cs:cs + 3]
        if k in box_list:
            return False
        else:
            return True
        
    def check_row(self, i, j, k): 
        row_list = self.number_table[i,:]
        if k in row_list:
            return False
        else:
            return True
        
    def check_column(self, i, j, k): 
        column_list = self.number_table[:,j]
        if k in column_list:
            return False
        else:
            return True