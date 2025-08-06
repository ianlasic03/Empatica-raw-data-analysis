import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

class EyeTracking():
    def __init__(self, input_file):
        self.input_file = input_file

    def plot_eye_direction(self, stopwatch, dx, dy, dz):
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


    def read_eyetracking(self, input_file):
        eye_data = {'timestamp': [],'stopwatch': [], 'ox': [], 'oy': [], 'oz': [],
                    'dx': [], 'dy': [], 'dz': [], 'status':[]}
        with open(input_file, mode='r') as file:
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
                #print(line)
            #print(len(eye_data['timestamp']))
        return eye_data


    """
    *In terms of viewing the aircraft cockpit*
    Transform eye tracking x-> left, y -> upwards, z -> outwards into
    aircraft x, y, z coordinates
    - Aircraft coordinates: x -> right side of aircraft, y -> upwards, z -> tail of aircraft
    """
    def eye_to_aircraft(self, eye_tracking):
        # This comes from transforming 
        eye_track_to_aircraft = np.array([[-1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, -1]
                                      ])
        eye_aircraft = np.matmul(eye_track_to_aircraft, eye_tracking)
        return eye_aircraft

    """
    Inputs to transformation matrices change because we need to 
    adjust for what access we are rotating around with respect to the aircraft coordinates
    Specifically:
    Roll -> T3(-phi)
    Pitch -> T1(theta)
    Yaw -> T2(-psi)
    """
    def rotation_transformations(self, phi, theta, psi):
        """T1 = np.array([[1, 0, 0],
                    [0, np.cos(phi), np.sin(phi)],
                    [0, -np.sin(phi), np.cos(phi)]
                    ])
        T2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]
                      ])
        T3 = np.array([[np.cos(psi), np.sin(psi), 0],
                      [-np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]
                      ])"""
        """
        Based on performing the right hand role on the axis and flight conventions:
        Conventions:
        roll: to the right is + , to the left is -
        pitch: upwards is +, downwards is -
        yaw: to the right is +, downwards is -

        Right hand curl based on aircraft coordinates (x right, y up, z tail)
        roll (phi) -> rotating around z axis (curl rule on z axis, left is +, right is - *need to reverse sign)
        pitch (theta) -> rotating around x axis (curl rule on x axis, up is +, down is -)
        yaw (psi) -> rotating around y axis (curl rule on y axis, left is +, right is - *need to reverse sign)
        """        
        T1 = np.array([[1, 0, 0],
                    [0, np.cos(-theta), np.sin(-theta)],
                    [0, -np.sin(-theta), np.cos(-theta)]
                    ])
        T2 = np.array([[np.cos(psi), 0, -np.sin(psi)],
                      [0, 1, 0],
                      [np.sin(psi), 0, np.cos(psi)]
                      ])
        T3 = np.array([[np.cos(-phi), np.sin(-phi), 0],
                      [-np.sin(-phi), np.cos(-phi), 0],
                      [0, 0, 1]
                      ])
        Rot_matrix = T3 @ T1 @ T2
        #print(Rot_matrix)
        return Rot_matrix


def main():
    eyetracking = EyeTracking('eye_tracking/eye_log_2.csv')
    eye_data = eyetracking.read_eyetracking(eyetracking.input_file)
    stopwatch = eye_data['stopwatch']
    forward_sanity_checks = [[0, 0, 0], [0,0,45], [0,0,-45], [0,45,0], [0,-45,0], [45,0,0], [-45,0,0]]
    # reversed dz because aircraft has z pointing towards tail
    eye_tracking_forward = np.array([0,0,1])
    eye_aircraft = eyetracking.eye_to_aircraft(np.array([0, 0, 1]))
    #print(eye_aircraft)
    for case in forward_sanity_checks:
        rot_mat = eyetracking.rotation_transformations(phi=np.radians(case[0]),theta=np.radians(case[1]),psi=np.radians(case[2]))
        print(f'xa, ya, za for {case}: ', rot_mat @ eye_aircraft)
    #rot_mat = eyetracking.rotation_transformations(phi=np.radians(0),theta=np.radians(45),psi=np.radians(0))

    #print()
    # Multiply dz by -1 to reverse direction
    
    #print("rot * eye tracking: ", np.matmul(rot_mat, eye_aircraft))
    #print("final mat adjusting dz: ", final_mat)
     
    ox = eye_data['ox']
    oy = eye_data['oy']
    oz = eye_data['oz']

    dx = eye_data['dx']
    dy = eye_data['dy']
    dz = eye_data['dz']
    """with open('eye_origin_direction.csv', 'w', newline='') as csvfile:
        f = ['ox','oy','oz','dx','dy','dz']
        writer = csv.DictWriter(csvfile, fieldnames=f)
        writer.writeheader()  

        for values in zip(ox, oy, oz, dx, dy, dz):
            row = dict(zip(f, values))
            writer.writerow(row)  

    eyetracking.plot_eye_direction(stopwatch, dx, dy, dz)"""
if __name__ == "__main__":
    main()