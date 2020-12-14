


class ANNConfig:

    def __init__(self, filename):
        lines = []
        with open(filename,'r') as f:
            lines = f.readlines()

        if lines:
            self.normalizeInput=bool(lines[0])
            self.learningRate = float(lines[1])
            self.epochs = int(lines[2])
            self.threshold = float(lines[3])
            self.momentum = float(lines[4])
            self.trainingType = lines[5].strip()
            self.layers = []
            self.numLayers = int(lines[6])
            i = 7
            for l in range(self.numLayers):
                self.layers.append(
                    {"inputs":int( lines[i+0] ), 
                    "neurons":int( lines[i+1] ), 
                    "activationFunction":lines[i+2].strip()}
                    )
                i = i+3
        else:
            print("error reading", filename)