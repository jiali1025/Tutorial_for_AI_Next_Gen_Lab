#  TODO: All models for process simulation


import torch.nn as nn
import torch
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def randomTrainingExample(time_data, input_data, output_data, s=5):
    L = time_data.size()[0]
    interval = 10

    time_train = time_data
    input_train = input_data[:, :]
    output_train = output_data[::s, :]
    system_0 = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float).cuda()

    return time_train, input_train, output_train, system_0


s = 20

dir = os.path.dirname(os.path.abspath(__file__))
data_dir = r"/home/pengfei/projects/neural_ode/TheoryGuidedRNN/dataset/CSTR_real_train"

data_time = np.load(os.path.join(data_dir, 'Reactor_ODE_1_data_t.npy'))
data_output = np.load(os.path.join(data_dir, 'Reactor_ODE_1_data_output.npy'))

data_mean = np.ndarray.mean(data_output, axis=0)
data_mean = np.array([0.3, 0.08, 0.4, 0.4, 0.35])

data_input = np.load(os.path.join(data_dir, 'Reactor_ODE_1_data_input.npy')) / data_mean
data_output = data_output / data_mean

data_time = torch.from_numpy(data_time).float().cuda()
data_output = torch.from_numpy(data_output).float().cuda()
data_input = torch.from_numpy(data_input).float().cuda()

data_dir_t = r"/home/pengfei/projects/neural_ode/TheoryGuidedRNN/dataset/CSTR_real_test"

data_time_t = np.load(os.path.join(data_dir_t, 'Reactor_ODE_1_data_t.npy'))
data_output_t = np.load(os.path.join(data_dir_t, 'Reactor_ODE_1_data_output.npy')) / data_mean
data_input_t = np.load(os.path.join(data_dir_t, 'Reactor_ODE_1_data_input.npy')) / data_mean

data_time_t = torch.from_numpy(data_time_t).float().cuda()
data_output_t = torch.from_numpy(data_output_t).float().cuda()
data_input_t = torch.from_numpy(data_input_t).float().cuda()

n_hidden = 10
n_para = 5

rnn = nn.LSTM(n_para, n_hidden, num_layers=1, proj_size=5).cuda()

# rnn.load_state_dict(torch.load(PATH))

# learning_rate = 0.001  # If you set this too high, it might explode. If too low, it might not learn
n_iters = 4000
print_every = 100
plot_every = 1000
every_graph = 10000

class LSTM():
    def __init__(self):
        pass

    def train(self):
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01, eps=0.0001)
        scheduler = StepLR(optimizer, step_size=3000, gamma=0.1)
        Loss_f = torch.nn.L1Loss(reduction='mean')

        current_loss = 0
        all_losses = []

        start = time.time()

        for iter in range(1, n_iters + 1):

            time_train, input_train, output_train, system_0 = randomTrainingExample(data_time, data_input, data_output,
                                                                                    s=s)

            # scale = torch.linspace(0.5, 1.5, L, dtype=torch.float, device='cuda')

            input_train = input_train.unsqueeze(1)
            output_train = output_train.unsqueeze(1)
            system_0 = torch.zeros([1, 1, 5]).cuda()
            cell_0 = torch.zeros([1, 1, n_hidden]).cuda()

            rnn.zero_grad()

            system_pred, (hn, cn) = rnn(input_train, (system_0, cell_0))

            loss = Loss_f(output_train[1:], system_pred[:-s:s])
            loss.backward()

            current_loss = current_loss + loss.item()

            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 10)
            optimizer.step()
            scheduler.step()

            # Add parameters' gradients to their values, multiplied by learning rate

            # p = 0.999 * p - (p.grad.data).clip(-2, 2)*learning_rate
            # para = rnn.parameters()
            # p.data.add_(p.grad.data, alpha=-learning_rate)

            # Print iter number, loss, name and guess

            if iter % print_every == 0:
                print('%d  (%s) %.4f %.4f' % (
                    iter, timeSince(start), current_loss / print_every, optimizer.param_groups[0]['lr']))
                all_losses.append(current_loss / plot_every)
                current_loss = 0

                time_train_t, input_train_t, output_train_t, system_0_t = randomTrainingExample(data_time_t, data_input_t,
                                                                                                data_output_t,
                                                                                                s=1)
                input_train_t = input_train_t.unsqueeze(1)
                output_train_t = output_train_t.unsqueeze(1)
                system_0_t = system_0_t.unsqueeze(1)
                cell_0_t = torch.zeros([1, 1, n_hidden]).cuda()

                system_pred_t, (hn, cn) = rnn(input_train_t, (system_0_t, cell_0_t))

                loss = Loss_f(output_train_t[1:], system_pred_t[:-1])

                print("test loss = {:.4f}".format(loss.item()))

                # for p in rnn.parameters():
                #     print(p.grad.data)

            # if iter % every_graph == 0:
            #     L = 100
        #
        time_train, input_train, output_train, system_0 = randomTrainingExample(data_time_t, data_input_t, data_output_t,
                                                                                s=s)
        input_train = input_train.unsqueeze(1)
        output_train = output_train.unsqueeze(1)
        system_0 = system_0.unsqueeze(1)
        cell_0 = torch.zeros([1, 1, n_hidden]).cuda()

        rnn.zero_grad()

        system, (hn, cn) = rnn(input_train, (system_0, cell_0))

        system = system.squeeze(1)
        output_train = output_train.squeeze(1)

        plt.figure()
        # t0 = range(time_train[::2])
        plt.plot(time_train.cpu().detach().numpy(), system.cpu().detach().numpy())
        plt.ylim(0, 1)
        plt.show()
        plt.figure()
        plt.plot(time_train[::s].cpu().detach().numpy(), output_train.cpu().detach().numpy())
        plt.ylim(0, 1)
        plt.show()

        return 0


train()
#
# save_dir = os.path.join(dir, 'output/')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# torch.save(rnn.state_dict(), os.path.join(save_dir, 'model.pth'))

