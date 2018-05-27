import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import matplotlib
import sys
import json

''' FUNCTION TO MONITOR THE LOGS FILES AND PLOT METRICS '''
parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, required=True)
parser.add_argument('--dynamic_plot', action='store_true')
args = parser.parse_args()
config_json = None
with open('../config.json', 'r') as f:
    config_json = json.load(f)


folder = '../plots'
file = open(args.log_file, 'r')
its = int(config_json['tot_iter'])

worker = int(config_json['nb_pods']) - 1


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def create_plot(worker, title='', x_title='', y_title=''):
    fig, ax = plt.subplots()
    ax.set_title(title)
    if worker == 0:
        plot, = ax.plot([], [])
    else:
        plot = []
        for i in range(worker):
            plt_tmp, = ax.plot([], [], label='worker_{}'.format(i))
            plot.append(plt_tmp)

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    return fig, ax, plot


def set_plot(plot, fig, axes, x, y):
    plot.set_xdata(x)
    plot.set_ydata(y)
    axes.relim()
    axes.autoscale_view(True, True, True)
    fig.canvas.draw()


try:

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
    weight_size = [[] for i in range(worker)]
    fig_loss, ax_loss, plt_loss = create_plot(
        worker, title='Train loss', x_title='Iterations', y_title='Regularized loss')
    fig_acc, ax_acc, plt_acc = create_plot(
        worker, title='Train accuracy', x_title='Iterations', y_title='Accuracy')
    fig_weight, ax_weight, plt_weight = create_plot(
        worker, title='Weights evolution through iterations', x_title='Iterations', y_title='Weights size')
    fig_its, ax_its, plt_its = create_plot(
        worker, title='Iteration against time', x_title='Time (s)', y_title='Iterations number')
    fig_loss_test, ax_loss_test, plt_loss_test = create_plot(
        0, title='Validation loss', x_title='Iterations', y_title='Regularized loss')
    fig_acc_test, ax_acc_test, plt_acc_test = create_plot(
        0, title='Validation accuracy', x_title='Train accuracy', y_title='Accuracy')
    ax_loss.legend()
    ax_acc.legend()
    ax_its.legend()
    ax_weight.legend()

    if args.dynamic_plot:
        print('[INFO] DYNAMIC PLOTTING MODE ON')
        plt.ion()
        plt.show(block=False)

except KeyboardInterrupt:
    print('[INFO] Exiting program')
    sys.exit()
try:
    while 1:
        if args.dynamic_plot:
            curr_time = time.time()
            if curr_time - start_time > 3:
                start_time = curr_time
                for idx in range(len(losses)):
                    idxs = ~np.isnan(losses[idx])
                    x = np.where(idxs)[0]
                    set_plot(plt_loss[idx], fig_loss,
                             ax_loss, x, losses[idx][idxs])
                    set_plot(plt_acc[idx], fig_acc, ax_acc, x, accs[idx][idxs])
                    set_plot(plt_its[idx], fig_its,
                             ax_its, times[idx], its[idx])
                    set_plot(plt_weight[idx], fig_weight,
                             ax_weight, its[idx], weight_size[idx])
                set_plot(plt_loss_test, fig_loss_test,
                         ax_loss_test, test_its, test_losses)
                set_plot(plt_acc_test, fig_acc_test,
                         ax_acc_test, test_its, test_accs)
                plt.draw()
                mypause(0.0001)
        is_first_line = file.tell() == 0
        line = file.readline()
        if is_first_line and line:
            print(line)
        if not line:
            time.sleep(1)
            continue

        # print(line)
        line_splitted = line.split(' ')
        if('worker' in line_splitted[0]):
            worker_nr = int(line_splitted[1][:-1])

            it = int(line_splitted[5])
            time_ = float(line_splitted[3])
            if last_its[worker_nr] < it:
                times[worker_nr].append(time_)
                its[worker_nr].append(it)
                last_its[worker_nr] = it
                weight_size[worker_nr].append(int(line_splitted[11]))

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
        elif('Started computing test metrics on the whole dataset' in line):
            print(line)
        elif('[TEST_FINAL]' in line):
            print(line)
            break
except KeyboardInterrupt:
    print('[INFO] Exiting program and saving plots')
    pass
finally:
    if not os.path.isdir(folder):
        os.mkdir(folder)

    for idx in range(len(losses)):
        idxs = ~np.isnan(losses[idx])
        x = np.where(idxs)[0]
        set_plot(plt_loss[idx], fig_loss,
                 ax_loss, x, losses[idx][idxs])
        set_plot(plt_acc[idx], fig_acc, ax_acc, x, accs[idx][idxs])
        set_plot(plt_its[idx], fig_its,
                 ax_its, times[idx], its[idx])
        set_plot(plt_weight[idx], fig_weight,
                 ax_weight, its[idx], weight_size[idx])
    set_plot(plt_loss_test, fig_loss_test,
             ax_loss_test, test_its, test_losses)
    set_plot(plt_acc_test, fig_acc_test,
             ax_acc_test, test_its, test_accs)

    fig_its.savefig('{}/speeds.png'.format(folder))
    fig_acc.savefig('{}/ACCS.png'.format(folder))
    fig_loss.savefig('{}/losses.png'.format(folder))
    fig_acc_test.savefig('{}/ACCS_test.png'.format(folder))
    fig_weight.savefig('{}/weights.png'.format(folder))
    fig_loss_test.savefig('{}/losses_test.png'.format(folder))
    file.close()
    print('[INFO] ended training')
