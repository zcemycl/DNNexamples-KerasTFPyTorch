from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1,box2):
    x11,y11,x12,y12 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
    x21,y21,x22,y22 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]
    
    interx1 =  torch.max(x11, x21)
    intery1 =  torch.max(y11, y21)
    interx2 =  torch.min(x12, x22)
    intery2 =  torch.min(y12, y22)
    
    #Intersection area
    inter_area = torch.clamp(interx2-interx1+1,min=0)*torch.clamp(intery2-intery1+1,min=0)

    #Union Area
    b1_area = (x12 - x11 + 1)*(y12 - y11 + 1)
    b2_area = (x22 - x21 + 1)*(y22 - y21 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def letterbox_image(img,inp_dim):
    imgw,imgh = img.shape[1],img.shape[0]
    w,h = inp_dim
    neww = int(imgw*min(w/imgw,h/imgh))
    newh = int(imgh*min(w/imgw,h/imgh))
    resized_img = cv2.resize(img,(neww,newh),interpolation=cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1],inp_dim[0],3),128)
    canvas[(h-newh)//2:(h-newh)//2+newh,(w-neww)//2:(w-neww)//2+neww,:] = resized_img
    
    return canvas

# Need explanations !!!!
def predict_transform(prediction,inp_dim,anchors,num_classes,CUDA=True):
    
    batch_size = prediction.size(0)
    stride = inp_dim//prediction.size(2)
    # change this, because it does not work 
    # at the scale of 52
#    grid_size = inp_dim // stride
    grid_size = prediction.size(2)
#    print(grid_size)
    bbox_attrs = 5+num_classes # 85
    num_anchors = len(anchors) # usually 3
    
#    print(prediction.size())
    
    prediction = prediction.view(batch_size,
                    bbox_attrs*num_anchors,
                    grid_size*grid_size)
    
#    print(prediction.size())
    prediction = prediction.transpose(1,2).contiguous()
#    print(prediction.size())
    prediction = prediction.view(batch_size,
                    grid_size*grid_size*num_anchors,
                    bbox_attrs)
#    print(prediction.size())
    anchors = [(a[0]/stride,a[1]/stride) for a in anchors]
    
    # sigmoid centre X Y and object confidence
    # please refer to those 4 equations
    # tx ty to sigmoid
    prediction[:,:,0]=torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1]=torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4]=torch.sigmoid(prediction[:,:,4])
    
    # center offsets cx cy
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid) # a[nxn]
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    # bounding box attributes
    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
        
    prediction[:,:,:2] += x_y_offset
    
    # log space transform height and width
    # exp tw and exp th
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
        
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    # e.g. pw exp tw
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    # class predictions
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    # multiply with the scale factor
    prediction[:,:,:4] *= stride
        
    return prediction

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def write_results(prediction,confidence,num_classes,nms_conf=.4):
    # filter out the predictions with probability of objectness
    # lower than the confidence .5
    conf_mask = (prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    # xmin,ymin and xmax,ymax
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    write = False
    
    for ind in range(batch_size):
        image_pred = prediction[ind]
        # confidence thresholding with NMS
        
        # max score for each label 
        # ind for max score
        max_conf,max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5],max_conf,max_conf_score)
        # output bbox properties, objectness, conf
        image_pred = torch.cat(seq,1)
        
        # filter the effect of mask
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        # 5+2 = 7 (4 bbox properties, obj, maxconf,maxconfind)
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue
        
        # Get the distinct classes detected in the image
        img_classes = unique(image_pred_[:,-1])
        
        # Iterate over each class
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
# output in the format of 
# index of the image in the batch
# 4 corner coordinates
# objectness
# score of class with maximum confidence
# index of that class
    
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names