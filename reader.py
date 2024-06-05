import numpy as np

class Reader:
    def __init__(self, grad_file):
        self.grad = self.parse_grad(grad_file)

    @staticmethod
    def parse_grad(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        grad_data = []
        
        for line in lines[2:]:
            data = line.split()
            atom = data[0]
            grad_x = float(data[1])
            grad_y = float(data[2])
            grad_z = float(data[3])
            grad_data.append([atom, grad_x, grad_y, grad_z])

        return np.array(grad_data, dtype=object)