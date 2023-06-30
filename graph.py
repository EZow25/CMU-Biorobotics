import numpy as np
import matplotlib.pyplot as plt
import transforms3d as t3d

iframe_rows = 4
iframe_cols = 4
def plot_image(pose):
    ax = plt.axes(projection ="3d")
    transform_matrix = get_transformation1(pose)
    
    x = []
    y = []
    z = []
    for i in range(-iframe_rows//2, (iframe_rows//2) + 1): # y axis
        for j in range(-iframe_cols, 1):                   # z axis
            m = np.array([0,i,j,1])
            m = np.matmul(transform_matrix, m)
            x.append(m[0])
            y.append(m[1])
            z.append(m[2])
    ax.scatter3D(x, y, z, color = "green")
    plt.title("Ultrasound Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def get_transformation1(pose):
    T_ru = np.eye(4)
    T_ru[0:3, 0:3] = t3d.quaternions.quat2mat([pose[6],pose[3],pose[4],pose[5]])
    T_ru[0:3, 3] = [pose[0], pose[1], pose[2]]
    
    return T_ru

plot_image([0.0,0.0,0.0,0,0.7071068,0,0.7071068])