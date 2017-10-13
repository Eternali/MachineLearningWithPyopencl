import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.io as spio
import os, sys, time


# input data file
INPUTFILE = '/home/conrad/Documents/cltests/international-airline-passengers.csv'


def load_data(filename):
    dataset = pandas.read_csv(filename, usecols=[1], engine='python', skipfooter=3)
    # plot the data for viewing
    plt.plot(dataset)
    plt.show()


def main():
    # fix random seed for reproducibility
    np.random.seed(7)


if __name__ == '__main__':
    main()

