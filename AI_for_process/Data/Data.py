#  TODO: provide data preprocessing and visualization modules here


import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class DataGen:
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(dir, 'dataset/')
        self.path = data_path

    def system_ODE_1(self, data_name='test_data', period=20, seed=0):
        def system(C, t, k=None, F=None, C_in_data=None):

            step = int(t)

            try:
                C_in = C_in_data[step]
            except:
                C_in = C_in_data[-1]

            dydt = [-k[0] * C[0] + F * (C_in[0] - C[0]),
                    k[0] * C[0] - k[1] * C[1] + F * (C_in[1] - C[1]),
                    k[1] * C[1] + F * (C_in[2] - C[2])]

            return dydt

        C_0 = [0, 0, 0]

        size = 50

        C_in = np.zeros([size * period, 3])

        np.random.seed(seed)
        for i in range(size):
            magn = np.random.rand(1)
            for C in C_in[i * period:(i + 1) * period]:
                C[0] = magn

            # C_in[100*i:100*(i+1)][0] = np.random.rand(1)*np.ones((100,1))

        k = [0.1, 0.02]
        F = 0.02
        t = np.linspace(0, size * period - 1, size * period)

        para = (k, F, C_in)

        sol = odeint(system, C_0, t, args=para)

        data = np.concatenate((np.expand_dims(t.astype('int'), axis=1), C_in, sol), axis=1)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, data_name), 'wb') as f:
            np.save(f, data)

        return data

    def visualize(self, data_name):

        data_path = os.path.join(self.path, data_name)
        data = np.load(data_path)

        t = data[:, 0]
        C_in = data[:, 1:4]
        C_out = data[:, 4:]

        fig, axs = plt.subplots(1, 2, figsize=(18, 5))

        axs[0].plot(t, C_out[:, 0], 'b', label='A(t)')
        axs[0].plot(t, C_out[:, 1], 'g', label='B(t)')
        axs[0].plot(t, C_out[:, 2], 'r', label='C(t)')
        axs[0].legend(loc='best')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('concentration')
        axs[0].set_title('output data')

        axs[1].plot(t, C_in[:, 0], 'b', label='A(t)')
        axs[1].plot(t, C_in[:, 1], 'g', label='B(t)')
        axs[1].plot(t, C_in[:, 2], 'r', label='C(t)')
        axs[1].legend(loc='best')
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('concentration')
        axs[1].set_title('input data')

        plt.show()
