import os
import numpy as np
from core.entities import MLEntityBase


class DataContainer(MLEntityBase):
    def __init__(self, dataset):
        super(DataContainer, self).__init__()
        self.DatasetName = dataset
        self.Data = None
        self.val_steps = 0
        self.test_steps = 0
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def build_generator(self, data, lookback, delay, min_index, max_index, shuffle, batch_size, step):
        if(max_index is None):
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randomint(min_index + lookback, max_index, size = batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
            samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                targets[j]= data[rows[j] + delay][1]
            yield samples, targets

    def generator(self, data, lookback, delay, min_index, max_index, shuffle, batch_size, step):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            #print(f'lookback:{lookback}, delay:{delay}, min_index{min_index}, max_index:{max_index}, shuffle{shuffle}, batch_size:{batch_size}, step: {step}')
            #print(f'Sample: {samples[0,0,0]} Target: {targets[0]}')
            yield samples, targets

    def prepare_data(self, sample_size):
        dir = os.path.join(self.Base_Directory, 'data', 'jena_climate')
        fname = os.path.join(dir, 'jena_climate_2009_2016.csv')
        with open(fname, 'r') as f:
            raw_data = f.read()
        f.close()
        all_lines = raw_data.split('\n')
        header = all_lines[0].split(',')
        lines = all_lines[1:]

        raw_data = np.zeros((len(lines), len(header) - 1))
        for i, line in enumerate(lines):
            values = [float(x) for x in line.split(',')[1:]]
            raw_data[i,:] = values

        mean = raw_data[:sample_size].mean(axis=0)
        regularized_data = raw_data - mean
        std = regularized_data[:sample_size].std(axis = 0)
        self.Data = regularized_data / std

        lookback =1440
        step =6
        delay = 144
        batch_size = 128
        min_index = 0
        max_index = 200000
        self.train_generator = self.generator(self.Data, lookback, delay, min_index, max_index, True, batch_size, step)
        min_index = 200001
        max_index = 300000
        self.val_steps = (max_index - min_index - lookback) // batch_size
        self.val_generator = self.generator(self.Data, lookback, delay, min_index, max_index, False, batch_size, step)
        min_index = 300001
        max_index = None
        self.test_steps = (len(self.Data) - min_index - lookback) // batch_size
        self.test_generator = self.generator(self.Data, lookback, delay, min_index, max_index, False, batch_size, step)

        # temp = data[:,1]
        # #plt.plot(range(len(temp)), temp)
        # #plt.show()
        # plt.figure(2)
        # plt.plot(range(14400), temp[:14400])
        # plt.show()
