from dataLoader import DataLoader
import numpy as np

class polarTest():
    def __init__(self, input_file):
        self.input_file = input_file
        self.RRis = []
        self.data_loader = DataLoader

    def load_RRis(self):
        with open(self.input_file, 'r') as file: 
            self.RRis = np.array([int(line.strip()) for line in file if line.strip()]) / 1000
        return self.RRis
    
    def calculate_HR(self):
        HR = self.data_loader.calculate_heartrate(self, self.RRis)
        return HR
    
    def calculate_SDNN(self):
        SDNN = self.data_loader.calculate_segmented_sdnn(self, self.RRis, 20)
        return SDNN
    
    def calculate_RMSSD(self):
        RMSSD = self.data_loader.calculate_RMSSD(self, self.RRis)
        return RMSSD
    
    def calculate_cont_SDNN(self):
        cont_SDNN = self.data_loader.smoothing_SDNN(self, self.RRis)
        return cont_SDNN
    
    def calculate_cont_RMSSD(self):
        cont_RMSSD = self.data_loader.smoothing_RMSSD(self, self.RRis)
        return cont_RMSSD
    
def main():
    # Initialize instance of class
    polar = polarTest("/Users/ianlasic/SISLfflighttest2/polar_RRis.txt")

    
    RRs = polarTest.load_RRis(polar) 
    print(RRs)
    HR = polarTest.calculate_HR(polar)
    SDNN = polarTest.calculate_SDNN(polar)
    RMSSD = polarTest.calculate_RMSSD(polar)
    #SDNN_cont = polarTest.calculate_cont_SDNN(polar)
    #RMSSD_cont = polarTest.calculate_cont_RMSSD(polar)

    print("RRis: ", np.mean(RRs)* 1000)
    print("HR: ", np.mean(HR))
    print("SDNN: ", SDNN * 1000)
    print("RMSSD: ", RMSSD * 1000) 
    #print("cont SDNN: ", SDNN_cont)
    #print("cont RMSSD: ", RMSSD_cont)


if __name__ == "__main__":
    main()