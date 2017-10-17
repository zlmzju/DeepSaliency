# -*- coding: utf-8 -*-
import caffe
import os
import h5py
import numpy as np
import cv2
def process(imageDir,gtDir,prefix,filelist,M,N):
    batchSize=500
    imageList=os.listdir(gtDir)
    imageData=np.zeros([batchSize,3,M,N]);
    inputMap=np.zeros([batchSize,1,M,N]);
    idx=0
    #image
    transformer = caffe.io.Transformer({'data': (1,3,M,N)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([104.00698793,  116.66876762,  122.67891434])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    for string in imageList:
        filename=string
        if filename[-1]=='\n':
            filename=filename[0:-1]
        if filename[-1]!='g':
            continue
        filename=filename[0:-4]+'.jpg'
        mapname=filename[0:-4]+'.png'
        filename=imageDir+filename
        mapname=gtDir+mapname
        image=transformer.preprocess('data', caffe.io.load_image(filename))
        #map
        map=cv2.imread(mapname,0)
        map=cv2.resize(map,(M,N))
        map[map>0]=1
        #imageData and inputMap
        imageData[idx%batchSize,:]=image
        inputMap[idx%batchSize,:]=map
        if (idx%batchSize)>=(batchSize-1) or idx==(len(imageList)-1):  #never reach >
            h5_filename=os.getcwd()+'/'+prefix+'%d.h5'%np.ceil(idx/batchSize)
            with h5py.File(h5_filename, 'w') as f:
                f['data'] = imageData[0:idx%batchSize+1,:]
                f['inputmap'] = inputMap[0:idx%batchSize+1,:]
            print h5_filename
            filelist.write(h5_filename+'\n')
        idx=idx+1
    return idx

M=500
N=500
dataDir='./MSRA1000/'
save_root=dataDir +'hdf5/'
if os.path.exists(save_root):
    shutil.rmtree(save_root)
    os.makedirs(save_root)
else:
    os.makedirs(save_root)
filelist=open(save_root+'train_list.txt','w')
listNum=process(dataDir+'Images/',dataDir+'Groundtruth/',save_root,filelist,M,N)
filelist.close()
