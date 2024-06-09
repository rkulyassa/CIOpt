class Writer:
    def __init__(self, geom_file):
        self.geom_file = geom_file
    
    def write_geom(self, geom_data):
        with open(self.geom_file, 'w'):
            pass