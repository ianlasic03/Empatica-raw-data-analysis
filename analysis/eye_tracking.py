import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def plot_eye_origin(stopwatch, ox, oy, oz):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    data = [ox, oy, oz]
    labels = ['ox', 'oy', 'oz']
    colors = ['black', 'blue', 'red']

    for i, ax in enumerate(axes):
        ax.plot(stopwatch, data[i], color=colors[i], label=labels[i])
        ax.set_title(f"{labels[i]} over time")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.legend()
        ax.grid(True)

    fig.suptitle("Eye Origin X, Y, Z Over Time", fontsize=16)
    fig.supxlabel("Time (s)")
    fig.supylabel("Eye Origin")

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()

def plot_eye_direction(stopwatch, dx, dy, dz):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    data = [dx, dy, dz]
    labels = ['dx', 'dy', 'dz']
    colors = ['black', 'blue', 'red']

    for i, ax in enumerate(axes):
        ax.plot(stopwatch, data[i], color=colors[i], label=labels[i])
        ax.set_title(f"{labels[i]} over time")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.legend()
        ax.grid(True)

    fig.suptitle("Eye direction X, Y, Z Over Time", fontsize=16)
    fig.supxlabel("Time (s)")
    fig.supylabel("Eye Origin")

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()

"""def plot_eye_orgin(stopwatch, ox, oy, oz):
    #fig, ax = plt.subplots(figsize=(12, 5))
    fig, ax = plt.subplots(1, 3)
    #plt.figure(figsize=(12, 5))
    ax[0].plot(stopwatch, ox, color='black', label='ox')
    ax[1].plot(stopwatch, oy, color='blue', label='oy')
    ax[2].plot(stopwatch, oz, color='red', label='oz')

    ax[0].xaxis.set_major_locator(MultipleLocator(100))
    ax[0].yaxis.set_major_locator(MultipleLocator(100))

    ax[1].xaxis.set_major_locator(MultipleLocator(100))
    ax[1].yaxis.set_major_locator(MultipleLocator(100))
    ax[2].xaxis.set_major_locator(MultipleLocator(100))
    ax[2].yaxis.set_major_locator(MultipleLocator(100))
    
    plt.xlabel("time (s)")
    plt.ylabel("eye origin")
    plt.title("eye origin x, y, z over time ")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()"""

def read_eyetracking():
    eye_data = {'timestamp': [],'stopwatch': [], 'ox': [], 'oy': [], 'oz': [],
                'dx': [], 'dy': [], 'dz': [], 'status':[]}
    with open('eye_tracking/eye_log_2.csv', mode='r') as file:
        csv_dict = csv.DictReader(file)
        for line in csv_dict:
            if line['status'] == 'OK':
                eye_data['timestamp'].append(line['time'])
                eye_data['stopwatch'].append(line['stopwatch'])
                eye_data['ox'].append(line['ox'])
                eye_data['oy'].append(line['oy'])
                eye_data['oz'].append(line['oz'])
                eye_data['dx'].append(line['dx'])
                eye_data['dy'].append(line['dy'])
                eye_data['dz'].append(line['dz'])
            print(line)
        print(len(eye_data['timestamp']))
    return eye_data

#def smooth_

def main():
    eye_data = read_eyetracking()
    stopwatch = eye_data['stopwatch']
    ox = eye_data['ox']
    oy = eye_data['oy']
    oz = eye_data['oz']

    dx = eye_data['dx']
    dy = eye_data['dy']
    dz = eye_data['dz']
    with open('eye_origin_direction.csv', 'w', newline='') as csvfile:
        f = ['ox','oy','oz','dx','dy','dz']
        writer = csv.DictWriter(csvfile, fieldnames=f)
        writer.writeheader()  # optional, writes the header row

        for values in zip(ox, oy, oz, dx, dy, dz):
            row = dict(zip(f, values))
            writer.writerow(row)  

    plot_eye_origin(stopwatch, ox, oy, oz)
    plot_eye_direction(stopwatch, dx, dy, dz)
if __name__ == "__main__":
    main()