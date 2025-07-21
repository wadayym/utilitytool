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
        self.set_order()
        print(self.number_table) 
        result = self.check(0)
        return result

    def set_order(self):
        self.work_table = np.copy(self.number_table)  
        zero_count = np.sum(self.input_table == 0)
        self.order_table = np.zeros((zero_count, 2), dtype=np.int32)
        for k in range(zero_count):         
            max = 0
            max_i = 0
            max_j = 0
            for i in range(9):
                for j in range(9):
                    if self.work_table[i][j] == 0:                       
                        count = self.count3(i, j)
                        if count > max:
                            max = count
                            max_i = i
                            max_j = j
            self.order_table[k][0] = max_i
            self.order_table[k][1] = max_j
            self.work_table[max_i][max_j] = 1  # Mark as processed
        #print("order_table:", self.order_table)

    def count3(self, i, j):
        return self.count_box(i, j) + self.count_row(i, j) + self.count_column(i, j)

    def count_box(self, i, j): 
        box_row = i // 3
        box_column = j // 3
        rs = box_row * 3
        cs = box_column * 3
        return np.count_nonzero(self.work_table[rs:rs + 3, cs:cs + 3])
        
    def count_row(self, i, j): 
        return np.count_nonzero(self.work_table[i,:])
        
    def count_column(self, i, j): 
        return np.count_nonzero(self.work_table[:,j])

    def check(self, n):
        #print("len:",len(self.order_table))
        if n >= len(self.order_table):
            return True
        i = self.order_table[n][0]
        j = self.order_table[n][1]
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