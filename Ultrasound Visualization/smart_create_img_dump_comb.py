import cv2
import numpy as np
import ipdb
import yaml
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import transforms3d as t3d

from itertools import accumulate
import operator


def template_gen(width, height):
    transforms = torch.zeros((height*width, 4, 4))
    lowx = 0 
    highx = int(height)
    lowy = int(0-width/2)
    highy = int(width/2)

    T2 = torch.tensor([  [0.0000000,  -1.0000000,  0.0000000, 0],
                     [1.0000000,  0.0000000,  0.0000000, 0],
                     [0.0000000,  0.0000000,  1.0000000, 0],
                     [0, 0, 0, 1]])
    sfy = 0.04/width
    sfx = 0.05/height

    z=0
    for i in range(lowx, highx, 1):
        for j in range(lowy,highy,1):
            T1 = torch.eye(4)
            T1[0:3,3] = torch.tensor([i*sfx, j*sfy,0])
            Tr = T2@T1
            # ipdb.set_trace()
            transforms[z] = torch.tensor(Tr)
            z+=1
    transforms=transforms.type(torch.DoubleTensor)

    return transforms
            
            

def location_extractor(image, T3, width, height, transforms):
    image=image.squeeze(0)
    locations_3d = torch.empty(image.shape[0], image.shape[1], 3)

    T3_ = T3.unsqueeze(0)
    T3_jumbo = T3_.repeat(510*522,1,1)
    prod = T3_jumbo@transforms
    XYZ = prod[:,0:3,3]*10000
    locations_3d = torch.reshape(XYZ, (height,width,3))

    return locations_3d



def get_transformation1( pose ):
        T_ru = np.eye(4)
        T_ru[0:3, 0:3] = t3d.quaternions.quat2mat([pose[6],pose[3],pose[4],pose[5]])
        T_ru[0:3, 3] = [pose[0], pose[1], pose[2]]
        return T_ru



def main():
    with open('pps_config.yml', 'r') as stream:
        entries = yaml.load(stream, Loader=yaml.SafeLoader)
    print(entries)

    width = entries['width']
    height = entries['height']

    transform_mats = template_gen(width, height)

    train_sweeps = len(entries['train'])
    val_sweeps = len(entries['val'])
    sidx = entries['sidx']
    eidx = entries['eidx']

    allposes=0
    flag=False
    for i in range(train_sweeps):
        pose_file = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['train'][i]+'/poses.npy'
        poses_ = np.load(pose_file)
        if(flag==False):
            flag=True
            allposes=poses_
        else:
            allposes = np.concatenate((allposes,poses_), axis=0)

    training_samples = allposes.shape[0]

    for i in range(val_sweeps):
        pose_file = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['val'][i]+'/poses.npy'
        poses_ = np.load(pose_file)
        allposes = np.concatenate((allposes,poses_), axis=0)

    val_samples = allposes.shape[0] - training_samples

    xmax = max(allposes[:,3])
    xmin = min(allposes[:,3])

    ymax = max(allposes[:,4])
    ymin = min(allposes[:,4])

    zmax = max(allposes[:,5])
    zmin = min(allposes[:,5])

    #training dump creation
    chkpt = eidx[0:train_sweeps]
    spts = sidx[0:train_sweeps]
    diff = list(map(operator.sub,chkpt,spts))
    cumsum = list(accumulate(diff, operator.add))

    z=0
    img_folder = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['train'][z]+'_image/'
    pose_file = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['train'][z]+'/poses.npy'
    poses = np.load(pose_file)
    
    poses[:,3] = (poses[:,3] - xmin)/(xmax-xmin)
    poses[:,4] = (poses[:,4] - ymin)/(ymax-ymin)
    poses[:,5] = (poses[:,5] - zmin)/(zmax-zmin)

    # ipdb.set_trace()

    flag2=0
    i=0
    while(i<training_samples):
        if(i<cumsum[z]):
            if(z==0):
                print(img_folder+'frame'+str(i+spts[z])+'.png')
                image =  cv2.imread(img_folder+'frame'+str(i+spts[z])+'.png', cv2.COLOR_BGR2GRAY)
                image = image[134:639,386:908,0:1]
                image = cv2.resize(image, (width, height))
                transform = transforms.ToTensor()
                image = transform(image).cuda()
                pose = poses[i][3:]
                T_rob2us = torch.tensor(get_transformation1(pose))

                if(i==0):
                    xyz = location_extractor(image, T_rob2us, width, height, transform_mats)
                    samples3d = xyz.unsqueeze(0)
                    
                else: 
                    xyz = location_extractor(image, T_rob2us, width, height, transform_mats)
                    xyz = xyz.unsqueeze(0)
                    samples3d = torch.cat((samples3d, xyz), dim=0)
            else:
                print(img_folder+'frame'+str(i-flag2+spts[z])+'.png')
                image =  cv2.imread(img_folder+'frame'+str(i-flag2+spts[z])+'.png', cv2.COLOR_BGR2GRAY)
                image = image[134:639,386:908,0:1]
                image = cv2.resize(image, (width, height))
                transform = transforms.ToTensor()
                image = transform(image).cuda()
                pose = poses[i-flag2][3:]
                T_rob2us = torch.tensor(get_transformation1(pose))

                xyz = location_extractor(image, T_rob2us, width, height, transform_mats)
                xyz = xyz.unsqueeze(0)
                samples3d = torch.cat((samples3d, xyz), dim=0)


        else:
            z+=1
            img_folder = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['train'][z]+'_image/'
            pose_file = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['train'][z]+'/poses.npy'
            poses = np.load(pose_file)
            
            poses[:,3] = (poses[:,3] - xmin)/(xmax-xmin)
            poses[:,4] = (poses[:,4] - ymin)/(ymax-ymin)
            poses[:,5] = (poses[:,5] - zmin)/(zmax-zmin)
            flag2=i
            continue
            
        i+=1

    tr_samples3d_np = samples3d.numpy()        
    # ipdb.set_trace()
    np.save('training_image_samples.npy', tr_samples3d_np)




    #validation dump creation
    chkpt = eidx[train_sweeps:train_sweeps+val_sweeps]
    spts = sidx[train_sweeps:train_sweeps+val_sweeps]
    diff = list(map(operator.sub,chkpt,spts))
    cumsum = list(accumulate(diff, operator.add))

    z=0
    img_folder = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['val'][z]+'_image/'
    pose_file = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['val'][z]+'/poses.npy'
    poses = np.load(pose_file)
    
    poses[:,3] = (poses[:,3] - xmin)/(xmax-xmin)
    poses[:,4] = (poses[:,4] - ymin)/(ymax-ymin)
    poses[:,5] = (poses[:,5] - zmin)/(zmax-zmin)
    # ipdb.set_trace()

    flag2=0
    i=0
    while(i<val_samples):
        if(i<cumsum[z]):
            if(z==0):
                print(img_folder+'frame'+str(i+spts[z])+'.png')
                image =  cv2.imread(img_folder+'frame'+str(i+spts[z])+'.png', cv2.COLOR_BGR2GRAY)
                image = image[134:639,386:908,0:1]
                image = cv2.resize(image, (width, height))
                transform = transforms.ToTensor()
                image = transform(image).cuda()
                pose = poses[i][3:]
                T_rob2us = torch.tensor(get_transformation1(pose))

                if(i==0):
                    xyz = location_extractor(image, T_rob2us, width, height, transform_mats)
                    samples3d = xyz.unsqueeze(0)
                    
                else: 
                    xyz = location_extractor(image, T_rob2us, width, height, transform_mats)
                    xyz = xyz.unsqueeze(0)
                    samples3d = torch.cat((samples3d, xyz), dim=0)
            else:
                print(img_folder+'frame'+str(i-flag2+spts[z])+'.png')
                # ipdb.set_trace()
                image =  cv2.imread(img_folder+'frame'+str(i-flag2+spts[z])+'.png', cv2.COLOR_BGR2GRAY)
                image = image[134:639,386:908,0:1]
                image = cv2.resize(image, (width, height))
                transform = transforms.ToTensor()
                image = transform(image).cuda()
                pose = poses[i-flag2][3:]
                T_rob2us = torch.tensor(get_transformation1(pose))

                xyz = location_extractor(image, T_rob2us, width, height, transform_mats)
                xyz = xyz.unsqueeze(0)
                samples3d = torch.cat((samples3d, xyz), dim=0)


        else:
            z+=1
            img_folder = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['val'][z]+'_image/'
            pose_file = '/home/ananya/ultranerf/Blue_gel_US/Lab_2/'+entries['val'][z]+'/poses.npy'
            poses = np.load(pose_file)
            
            poses[:,3] = (poses[:,3] - xmin)/(xmax-xmin)
            poses[:,4] = (poses[:,4] - ymin)/(ymax-ymin)
            poses[:,5] = (poses[:,5] - zmin)/(zmax-zmin)
            flag2=i
            # ipdb.set_trace()
            continue
            
        i+=1

    val_samples3d_np = samples3d.numpy() 
    np.save('val_image_samples.npy', val_samples3d_np)
    # ipdb.set_trace()



if __name__ == "__main__":
    main()
