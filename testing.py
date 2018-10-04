import os
import numpy as np
import pandas as pd

DATA_PATH="C:\\datasets\\CUB_200_2011\\CUB_200_2011\\"

def main():
    arr = np.arange(1500).reshape((10,50,3))
    arr = np.transpose(arr,(0,2,1))
    print(arr)

if __name__ == '__main__':
    main()
