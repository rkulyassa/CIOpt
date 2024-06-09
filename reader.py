import numpy as np

class Reader:
    def __init__(self, grad_file, geom_file):
        self.grad_file = grad_file
        self.geom_file = geom_file
        # self.grad = self.parse_grad(grad_file)
        # self.geom = self.parse_geom(geom_file)

    def parse_grad(self, grad_file = None):
        grad_file = grad_file or self.grad_file
        with open(grad_file, 'r') as file:
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
    
    def parse_geom(self, geom_file = None):
        geom_file = geom_file or self.geom_file