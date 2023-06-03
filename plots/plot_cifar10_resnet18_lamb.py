import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme(style="darkgrid")

if not os.path.exists('figures'):
    os.mkdir('figures')

def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def smoothing(array, width):
    length = len(array)
    output = np.zeros([length], dtype=float)

    ind_begin = 0
    for i in range(length):
        ind_end = i + 1
        if ind_end > width:
            ind_begin = ind_end - width
        output[i] = array[ind_begin:ind_end].mean()
    return output

ROOT = 'csv_data'
rolling_step = 1
smooth_rate = 0.9

csgd64_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.0035-0.0-0.1-0.0-0-6000-6000-666-False.csv')
csgd512_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.01-0.0-0.1-0.0-0-6000-6000-666-False.csv')
dsgd64_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-64-ring-random-16-ResNet18_M-1-0.0035-0.0-0.1-0.0-0-6000-6000-666-False.csv')
dsgd512_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-512-ring-random-16-ResNet18_M-1-0.01-0.0-0.1-0.0-0-6000-6000-666-False.csv')

dsgd64_seed666_data = pd.read_csv(dsgd64_seed666_path)
x_data = list(dsgd64_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd64_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-LAMB_1024(lr=0.0035)')

dsgd512_seed666_data = pd.read_csv(dsgd512_seed666_path)
x_data = list(dsgd512_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd512_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-LAMB_8192(lr=0.01)')


csgd64_seed666_data = pd.read_csv(csgd64_seed666_path)
x_data = list(csgd64_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd64_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-LAMB_1024(lr=0.0035)')

csgd512_seed666_data = pd.read_csv(csgd512_seed666_path)
x_data = list(csgd512_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd512_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-LAMB_8192(lr=0.01)')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.66,0.96)
plt.legend(loc='lower right', fontsize=18, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'figures/fig_cifar10_resnet18_lamb001.pdf', format='pdf', bbox_inches='tight')
plt.close()


rolling_step = 1
smooth_rate = 0.9

csgd64_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.007-0.0-0.1-0.0-0-6000-6000-666-False.csv')
csgd512_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.02-0.0-0.1-0.0-0-6000-6000-666-False.csv')
dsgd64_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-64-ring-random-16-ResNet18_M-1-0.007-0.0-0.1-0.0-0-6000-6000-666-False.csv')
dsgd512_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-512-ring-random-16-ResNet18_M-1-0.02-0.0-0.1-0.0-0-6000-6000-666-False.csv')

dsgd64_seed666_data = pd.read_csv(dsgd64_seed666_path)
x_data = list(dsgd64_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd64_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-LAMB_1024(lr=0.007)')

dsgd512_seed666_data = pd.read_csv(dsgd512_seed666_path)
x_data = list(dsgd512_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd512_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-LAMB_8192(lr=0.02)')


csgd64_seed666_data = pd.read_csv(csgd64_seed666_path)
x_data = list(csgd64_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd64_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-LAMB_1024(lr=0.007)')

csgd512_seed666_data = pd.read_csv(csgd512_seed666_path)
x_data = list(csgd512_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd512_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-LAMB_8192(lr=0.02)')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.66,0.96)
plt.legend(loc='lower right', fontsize=18, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'figures/fig_cifar10_resnet18_lamb002.pdf', format='pdf', bbox_inches='tight')
plt.close()


rolling_step = 1
smooth_rate = 0.9

csgd64_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.0105-0.0-0.1-0.0-0-6000-6000-666-False.csv')
csgd512_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.03-0.0-0.1-0.0-0-6000-6000-666-False.csv')
dsgd64_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-64-ring-random-16-ResNet18_M-1-0.0105-0.0-0.1-0.0-0-6000-6000-666-False.csv')
dsgd512_seed666_path = os.path.join(ROOT, 'lamb_jupyter-chenkaixuan_CIFAR10s56-512-ring-random-16-ResNet18_M-1-0.03-0.0-0.1-0.0-0-6000-6000-666-False.csv')

dsgd64_seed666_data = pd.read_csv(dsgd64_seed666_path)
x_data = list(dsgd64_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd64_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-LAMB_1024(lr=0.0105)')

dsgd512_seed666_data = pd.read_csv(dsgd512_seed666_path)
x_data = list(dsgd512_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd512_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-LAMB_8192(lr=0.03)')


csgd64_seed666_data = pd.read_csv(csgd64_seed666_path)
x_data = list(csgd64_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd64_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-LAMB_1024(lr=0.0105)')

csgd512_seed666_data = pd.read_csv(csgd512_seed666_path)
x_data = list(csgd512_seed666_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd512_seed666_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-LAMB_8192(lr=0.03)')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.66,0.96)
plt.legend(loc='lower right', fontsize=18, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'figures/fig_cifar10_resnet18_lamb003.pdf', format='pdf', bbox_inches='tight')
plt.close()

