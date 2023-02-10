import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpmath
from matplotlib import gridspec
ann_size = 19
dot_area = 70
line_wid = 3
markevery_num = 3000
markersize_num = 9
latencys1 = pd.read_csv('data.csv')
plt.figure(figsize=(12, 5))
lines = []
labels = []
mark = ['->', '--^', '-.s', ':*']
colors = ['r', 'b', 'g', 'black']
i = -1
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.5, 2])


data1 = latencys1.loc[:1999, ["lane"]].values*1000
data2 = latencys1.loc[:1999, ["detection"]].values*1000
data3 = latencys1.loc[:1999, ["light"]].values*1000
data4 = latencys1.loc[:1999, ["sign"]].values*1000

cum = range(1,2001)
cum_percent = [x / 2000 for x in cum]
index_1 = np.where(np.array(cum_percent) > 0.9)[0][0]
index_2 = np.where(np.array(cum_percent) > 0.99)[0][0]
if 1==1:
    plt.subplot(gs[0])
    if 1==1:
        label_name = 'max(max(3,4)+2,1)'
        i = 0

        aaa=np.array([max(data3[i],data4[i]) for i in range(len(data3))]) + data2
        data=np.array([max(aaa[i],data1[i]) for i in range(len(data1))]).reshape(-1).tolist()
        data.sort()
        line = plt.plot(data, cum_percent, mark[i], label=label_name, color=colors[i], markevery=markevery_num, linewidth=line_wid, markersize=markersize_num)

        line = plt.scatter(data[index_1], cum_percent[index_1], color=colors[i], s=dot_area)

        #plt.annotate(r"$90^{th}$",
        #             xy=(latency[column].values[index_1]-65, latency['cum_percent'].values[index_1]-0.09),
        #             textcoords='data', color=colors[i], fontsize=ann_size)
        line = plt.scatter(data[index_2], cum_percent[index_2], color=colors[i], s=dot_area)
        #plt.annotate(r"$99^{th}$",
        #             xy=(latency[column].values[index_2]-500, latency['cum_percent'].values[index_2]+0.007),
        #             textcoords='data', color=colors[i], fontsize=ann_size)

        lines.append(line)
        labels.append(label_name)

    plt.title("(a)", y=-0.33, fontsize=ann_size+4)
    plt.xlabel("ms", fontsize=ann_size)
    plt.ylabel("CDF", fontsize=ann_size, x=0.90)
    plt.xticks(fontsize=ann_size)
    plt.yticks(fontsize=ann_size)
    plt.gca()
    plt.grid(b=True, which='major', color='gray', linestyle=':' )
    plt.grid(b=True, which='minor', color='gray', linestyle=(0, (1, 10)))
    plt.legend(loc=4, fontsize=ann_size-2)
    plt.xscale('log')


    plt.xlim(60, 90)
    plt.ylim(0, 1.03)

if 2==2:
    plt.subplot(gs[1])
    i = -1
    if 21==21:
        i = i + 1
        label_name = '1'
        data=data1.reshape(-1).tolist()
        data.sort()
        line = plt.plot(data, cum_percent, mark[i], label=label_name, color=colors[i], markevery=markevery_num, linewidth=line_wid, markersize=markersize_num)

        line = plt.scatter(data[index_1], cum_percent[index_1], color=colors[i], s=dot_area)

        #plt.annotate(r"$90^{th}$",
        #             xy=(latency[column].values[index_1]-65, latency['cum_percent'].values[index_1]-0.09),
        #             textcoords='data', color=colors[i], fontsize=ann_size)
        line = plt.scatter(data[index_2], cum_percent[index_2], color=colors[i], s=dot_area)
        #plt.annotate(r"$99^{th}$",
        #             xy=(latency[column].values[index_2]-500, latency['cum_percent'].values[index_2]+0.007),
        #             textcoords='data', color=colors[i], fontsize=ann_size)

        lines.append(line)
        labels.append(label_name)

    if 22==22:
        i = i + 1
        label_name = 'max(3,4)+2'
        data=(np.array([max(data3[i],data4[i]) for i in range(len(data3))]) + data2).reshape(-1).tolist()
        data.sort()
        line = plt.plot(data, cum_percent, mark[i], label=label_name, color=colors[i], markevery=markevery_num, linewidth=line_wid, markersize=markersize_num)

        line = plt.scatter(data[index_1], cum_percent[index_1], color=colors[i], s=dot_area)

        #plt.annotate(r"$90^{th}$",
        #             xy=(latency[column].values[index_1]-65, latency['cum_percent'].values[index_1]-0.09),
        #             textcoords='data', color=colors[i], fontsize=ann_size)
        line = plt.scatter(data[index_2], cum_percent[index_2], color=colors[i], s=dot_area)
        #plt.annotate(r"$99^{th}$",
        #             xy=(latency[column].values[index_2]-500, latency['cum_percent'].values[index_2]+0.007),
        #             textcoords='data', color=colors[i], fontsize=ann_size)

        lines.append(line)
        labels.append(label_name)


    plt.title("(b)", y=-0.33, fontsize=ann_size+4)
    plt.xlabel("ms", fontsize=ann_size)
    plt.ylabel("CDF", fontsize=ann_size, x=0.90)
    plt.xticks(fontsize=ann_size)
    plt.yticks(fontsize=ann_size)
    plt.gca()
    plt.grid(b=True, which='major', color='gray', linestyle=':')
    plt.grid(b=True, which='minor', color='gray', linestyle=(0, (1, 10)))
    plt.legend(loc=4, fontsize=ann_size-2)
    plt.xscale('log')


    plt.xlim(20, 100)
    plt.ylim(0, 1.03)





if 3==3:
    plt.subplot(gs[2])
    i = -1
    if 21==21:
        i = i + 1
        label_name = '3'
        data=data3.reshape(-1).tolist()
        data.sort()
        line = plt.plot(data, cum_percent, mark[i], label=label_name, color=colors[i], markevery=markevery_num, linewidth=line_wid, markersize=markersize_num)

        line = plt.scatter(data[index_1], cum_percent[index_1], color=colors[i], s=dot_area)

        #plt.annotate(r"$90^{th}$",
        #             xy=(latency[column].values[index_1]-65, latency['cum_percent'].values[index_1]-0.09),
        #             textcoords='data', color=colors[i], fontsize=ann_size)
        line = plt.scatter(data[index_2], cum_percent[index_2], color=colors[i], s=dot_area)
        #plt.annotate(r"$99^{th}$",
        #             xy=(latency[column].values[index_2]-500, latency['cum_percent'].values[index_2]+0.007),
        #             textcoords='data', color=colors[i], fontsize=ann_size)

        lines.append(line)
        labels.append(label_name)

    if 22==22:
        i = i + 1
        label_name = '4'
        data=data4.reshape(-1).tolist()
        data.sort()
        line = plt.plot(data, cum_percent, mark[i], label=label_name, color=colors[i], markevery=markevery_num, linewidth=line_wid, markersize=markersize_num)

        line = plt.scatter(data[index_1], cum_percent[index_1], color=colors[i], s=dot_area)

        #plt.annotate(r"$90^{th}$",
        #             xy=(latency[column].values[index_1]-65, latency['cum_percent'].values[index_1]-0.09),
        #             textcoords='data', color=colors[i], fontsize=ann_size)
        line = plt.scatter(data[index_2], cum_percent[index_2], color=colors[i], s=dot_area)
        #plt.annotate(r"$99^{th}$",
        #             xy=(latency[column].values[index_2]-500, latency['cum_percent'].values[index_2]+0.007),
        #             textcoords='data', color=colors[i], fontsize=ann_size)

        lines.append(line)
        labels.append(label_name)


    plt.title("(b)", y=-0.33, fontsize=ann_size+4)
    plt.xlabel("ms", fontsize=ann_size)
    plt.ylabel("CDF", fontsize=ann_size, x=0.90)
    plt.xticks(fontsize=ann_size)
    plt.yticks(fontsize=ann_size)
    plt.gca()
    plt.grid(b=True, which='major', color='gray', linestyle=':')
    plt.grid(b=True, which='minor', color='gray', linestyle=(0, (1, 10)))
    plt.legend(loc=4, fontsize=ann_size-2)
    plt.xscale('log')


    plt.xlim(10, 100)
    plt.ylim(0, 1.03)


plt.subplots_adjust(left=0.075, right=0.97, top=0.95, bottom=0.25, wspace=0.29, hspace=0)
plt.show()
