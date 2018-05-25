import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import os
file = open('../logs/log.txt', 'r')
its = 10000
worker = int(sys.argv[1])


def create_plot(worker):
    fig, ax = plt.subplots()
    if worker == 0:
        plot, = ax.plot([2,3], [2,5])
    else:
        plot = []
        for i in range(worker):
            plt_tmp, = ax.plot([2,3], [2,5], label='worker_{}'.format(i))
            plot.append(plt_tmp)

    return fig, ax, plot


def set_plot(plot,fig,axes, x, y):
    plot.set_xdata(x)
    plot.set_ydata(y)
    axes.relim()
    axes.autoscale_view(True,True,True)
    fig.canvas.draw()

losses = [np.zeros(its) + float('nan') for x in range(worker)]
accs = [np.zeros(its) + float('nan') for x in range(worker)]
count = np.zeros(its)
test_losses = []
test_accs = []
test_its = []
line = file.readline()
times = [[] for x in range(worker)]
its = [[] for x in range(worker)]
last_its = [-1] * worker
start_time = time.time()
fig_loss, ax_loss, plt_loss = create_plot(worker)
fig_acc, ax_acc, plt_acc = create_plot(worker)
fig_its, ax_its, plt_its = create_plot(worker)
fig_loss_test, ax_loss_test, plt_loss_test = create_plot(0)
fig_acc_test, ax_acc_test, plt_acc_test = create_plot(0)

ax_loss.legend()
ax_acc.legend()
ax_its.legend()
folder = '../plts'
while 1:

    curr_time = time.time()
    if curr_time - start_time > 5:
        start_time = curr_time
        for idx in range(len(losses)):
            idxs = ~np.isnan(losses[idx])
            x = np.where(idxs)[0]
            set_plot(plt_loss[idx],fig_loss,ax_loss, x, losses[idx][idxs])
            set_plot(plt_acc[idx],fig_acc,ax_acc, x, accs[idx][idxs])
            set_plot(plt_its[idx],fig_its,ax_its, times[idx], its[idx])

        set_plot(plt_loss_test,fig_loss_test,ax_loss_test, test_its, test_losses)
        set_plot(plt_acc_test,fig_acc_test,ax_acc_test, test_its, test_accs)

        if not os.path.isdir(folder):
            os.mkdir(folder)

        fig_its.savefig('{}/speeds.png'.format(folder))
        fig_acc.savefig('{}/ACCS.png'.format(folder))
        fig_loss.savefig('{}/losses.png'.format(folder))

        fig_acc_test.savefig('{}/ACCS_test.png'.format(folder))
        fig_loss_test.savefig('{}/losses_test.png'.format(folder))


    line = file.readline()
    if not line:
        time.sleep(1)
        continue

    #print(line)
    line_splitted = line.split(' ')
    if('worker' in line_splitted[0]):
        worker_nr = int(line_splitted[1][:-1])

        it = int(line_splitted[5])
        time_ = float(line_splitted[3])
        if last_its[worker_nr] < it:
            times[worker_nr].append(time_)
            its[worker_nr].append(it)
            last_its[worker_nr] = it

        if(losses[worker_nr][it] != losses[worker_nr][it]):
            losses[worker_nr][it] = float(line_splitted[7])
            accs[worker_nr][it] = float(line_splitted[9])
        else:
            losses[worker_nr][it] += float(line_splitted[7])
            accs[worker_nr][it] += float(line_splitted[9])
    elif('[TEST]' in line_splitted[0]):
        test_losses.append(float(line_splitted[6]))
        test_accs.append(float(line_splitted[8]))
        test_its.append(float(line_splitted[4]))

