import os
import matplotlib.pyplot as plt

from practicalml.core.entities import MLEntity


class DataContainer(MLEntity):
    def __init__(self, dataset):
        self.Dataset = dataset
        self.val_steps = 0
        self.test_steps = 0
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def prepare_data(self, sample_size):
        dir = os.path.join(self.Base_Directory, 'data', 'jena_climate')
        fname = os.path.join(dir, 'jena_climate_2009_2016.csv')
        with open(fname, 'r') as f:
            raw_data = f.read()
        f.close()
        all_lines = raw_data.split('\n')
        header = all_lines[0].split(',')
        lines = all_lines[1:]

        data = np.zeros((len(lines), len(header) - 1))
        for i, line in enumerate(lines):
            values = [float(x) for x in line.split(',')[1:]]
            data[i,:] = values

        mean = data[:sample_size].mean(axis=0)
        regularized_data = data - mean
        std = regularized_data[:sample_size].std(axis = 0)
        std_data = regularized_data / std

        lookback =1440
        step =6
        delay = 144
        batch_size = 128
        min_index = 0
        max_index = 200000
        self.train_generator = self.build_generator(std_data, lookback, delay, min_index, max_index,
                                          shuffle=False, batch_size=batch_size, step= step)
        min_index = 200001
        max_index = 300000
        self.val_steps = max_index - min_index - lookback
        self.val_generator = self.build_generator(std_data, lookback, delay, min_index, max_index,
                                          shuffle=False, batch_size=batch_size, step= step)
        min_index = 300001
        max_index = None
        self.test_steps = len(std_data) - min_index - lookback
        self.test_generator = self.build_generator(std_data, lookback, delay, min_index, max_index,
                                          shuffle=False, batch_size=batch_size, step= step)

        # temp = data[:,1]
        # #plt.plot(range(len(temp)), temp)
        # #plt.show()
        # plt.figure(2)
        # plt.plot(range(14400), temp[:14400])
        # plt.show()
