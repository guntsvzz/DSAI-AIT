import numpy as np
import torch 
class snake_dataset():
    
    def __init__(self, num_sample = 1000):
        self.pi = np.pi
        self.num_sample = num_sample
        self.r = torch.randn(self.num_sample,1)
        self.theta = torch.FloatTensor(self.num_sample,1).uniform_(0,2*self.pi)
        self.a = 10 + self.r
        self.data = torch.empty(self.num_sample,2)
        #self.label = 0
        # self.Y = torch.empty(self.num_sample,1)
        for i in range(self.num_sample):
            if  0.5* self.pi <= self.theta[i] and self.theta[i] <= (3/2) * self.pi :
                self.a = 10 + self.r[i]
                self.x_data = self.a * torch.cos(self.theta[i])
                self.y_data = (self.a * torch.sin(self.theta[i])) + 10
                self.data[i,0] = self.x_data
                self.data[i,1] = self.y_data

            else:
                self.a = 10 + self.r[i]
                self.x_data = self.a * torch.cos(self.theta[i])
                self.y_data = (self.a * torch.sin(self.theta[i])) - 10
                self.data[i,0] = self.x_data
                self.data[i,1] = self.y_data

        self.len = self.data.shape[0]

    def __getitem__(self, index):
        # return (self.X[index], self.Y[index])
        return self.data[index]

    def __len__(self):
        return self.len