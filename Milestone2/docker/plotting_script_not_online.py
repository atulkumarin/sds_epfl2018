import matplotlib.pyplot as plt
import numpy as np
import sys
file = open('../data/out.txt', 'r')
its = 10000
worker = int(sys.argv[1])

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
while line:

    where = file.tell()
    line = file.readline()

    print(line)
    line_splitted = line.split(' ')
    if('worker' in line_splitted[0]):
        worker_nr = int(line_splitted[1][:-1])

        it = int(line_splitted[5])
        time = float(line_splitted[3])
        if last_its[worker_nr] < it:
            times[worker_nr].append(time)
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


fig_loss, ax_loss = plt.subplots()
fig_acc, ax_acc = plt.subplots()
fig_its, ax_its = plt.subplots()
for idx in range(len(losses)):
    idxs = ~np.isnan(losses[idx])
    x = np.where(idxs)[0]
    ax_loss.plot(x, losses[idx][idxs], label='worker_{}'.format(idx))
    ax_acc.plot(x, accs[idx][idxs], label='worker_{}'.format(idx))
    ax_its.plot(times[idx], its[idx], label='worker_{}'.format(idx))
    print('OK')


ax_loss.legend()
ax_acc.legend()
ax_its.legend()
fig_its.savefig('speeds.png')
fig_acc.savefig('ACCS.png')
fig_loss.savefig('losses.png')

plt.close(fig_loss)
plt.close(fig_acc)

fig_loss, ax_loss = plt.subplots()
fig_acc, ax_acc = plt.subplots()
ax_loss.plot(test_its, test_losses)
ax_acc.plot(test_its, test_accs)
fig_acc.savefig('ACCS_test.png')
fig_loss.savefig('losses_test.png')
plt.close(fig_loss)
plt.close(fig_acc)

plt.close(fig_loss)
plt.close(fig_acc)
