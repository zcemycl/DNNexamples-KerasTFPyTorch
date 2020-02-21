from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *
import numpy as np

def parse_cfg(cfgfile):
    file = open(cfgfile,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x)>0] # remove ''
    lines = [x for x in lines if x[0] != '#'] # remove comments
    lines = [x.rstrip().lstrip() for x in lines] # remove empty spaces from left and right
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

DIRcfg = "D:/DreamAI/YOLOforCardFace/YOLOfromScratch/cfg/yolov3.cfg"
blocks = parse_cfg(DIRcfg)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()
        
class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index,x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        # conv+batchnorm+act group
        if (x["type"] == "convolutional"):
            # get info about batch norm layer
            # This affects bias in convolutional layer
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            # get info about convolutional layer
            filters,padding = int(x["filters"]),int(x["pad"])
            kernel_size,stride = int(x["size"]),int(x["stride"])
            
            pad = (kernel_size-1)//2 if padding else 0
            
            # add convolutional layer and batch norm layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),
                              conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),
                                  bn)
                
            # get info about activation
            # either linear or leaky-relu
            activation = x["activation"] 
            if activation == "leaky":
                activn = nn.LeakyReLU(.1, inplace=True)
                module.add_module("leaky_{0}".format(index),
                                  activn)
        
        # Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2,
                                   mode="nearest")
            module.add_module("upsample_{}".format(index),
                              upsample)
        
        # Route layer (used to concatenate)
        elif (x["type"] == "route"):
            # split the layer no, e.g. -1,61
            # -n last nth layer, n nth layer
            x["layers"] = x["layers"].split(',')
            # Start of a route
            # get at least one number
            start = int(x["layers"][0]) 
            # end if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # for the case of the first no is 61
            # 61-63=-2
            if start > 0: 
                start -= index
            if end > 0:
                end -= index
                
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),
                              route)
            # to deal with end = -1 else 61
            # output filter number for next layer
            if end < 0:
                filters = output_filters[index+start]+output_filters[index+end]
            else:
                filters = output_filters[index+start]
        
        # Shortcut for skip connection
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),
                              shortcut)
            
        # YOLO detection layer for Feature Pyramid
        # Structure (FPS)
        elif (x["type"] == "yolo"):
            # e.g. to separate 6,7,8 
            mask = x["mask"].split(",")
            # turn them as int
            mask = [int(x) for x in mask]
            
            # to separate 9 anchors
            # 10,13, 16,30, 33,23, 30,61, 
            # 62,45, 59,119, 116,90, 156,198
            # 373,326 
            # if mask = 6,7,8
            # then only 116,90, 156,198 
            # 373,326 are used.
            anchors = x["anchors"].split(",")
            # turn as int
            anchors = [int(a) for a in anchors]
            # every 2 elements as one anchor
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index),
                              detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    # net_info is training settings
    return (net_info,module_list)

#print(create_modules(blocks))

# build the darknet with forward and load weights
class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,self.module_list = create_modules(self.blocks)
        
    def forward(self,x,CUDA):
        modules = self.blocks[1:]
        outputs = {}
        
        write = 0
        # iterate according to the blocks
        for i,module in enumerate(modules):
            module_type = (module["type"])
            
            # If convolution or upsample
            # direct forward
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                
            elif module_type == "route":
                # get last n-frame or n-th frame
                layers = module["layers"]
                # turn into number
                layers = [int(a) for a in layers]
                # pos then reduce last n-th frame
                if (layers[0]) > 0:
                    layers[0] -= i
                # only one layer, no cat
                if len(layers) == 1:
                    x = outputs[i+(layers[0])]
                else: 
                    # if second number is written 
                    # in pos, reduce to last n-th frame
                    if (layers[1]) > 0:
                        layers[1] -= i
                        
                    map1 = outputs[i+layers[0]]
                    map2 = outputs[i+layers[1]]
                    x = torch.cat((map1,map2),1)
            
            # adding two layers' outputs 
            elif module_type == "shortcut":
                # obtain from index 
                # negative: last n-th output
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                
            elif module_type == "yolo":
                # get anchors e.g. (10,13)
                anchors = self.module_list[i][0].anchors
                # get the input dimensions
                inp_dim = int(self.net_info["height"])
                
                # Get the number of classes
                num_classes = int(module["classes"])
                
                # Transform
                x = x.data
                x = predict_transform(x,inp_dim,
                        anchors,num_classes,CUDA)
                # store detections with different scales
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x),1)
                    
            outputs[i] = x
            
        return detections
    
    def load_weights(self,weightfile):
        fp = open(weightfile,"rb")
        # First 5 values are header info
        header = np.fromfile(fp,dtype=np.int32,count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        weights = np.fromfile(fp, dtype=np.float32)
        
        # counter for no of weights
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            
            # Just load weights for conv,batchnorm
            # otherwise ignore
            # There are two types of conv, batchnorm = True
            # with bias, else no bias
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try: 
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                
                # batch norm weights
                if (batch_normalize):
                    bn = model[1]
                    
                    # Get the no. of weights of BatchNorm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    # Load the weights (biases and weights)
                    # mean and var
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    
                    # Reshape the loaded weights into 
                    # the dims of untrained model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    
                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                    
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                
                # conv weights
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

        
            
# assume you are using gpu
# just to avoid memory problem
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = Darknet(DIRcfg)
#model.load_weights("yolov3.weights")
#model = model.to(device)
#inp = get_test_input().to(device)
#with torch.no_grad():
#    pred = model(inp, torch.cuda.is_available())
#    print(pred)
#    print(pred.size())
##    del pred
##    del inp
#    torch.cuda.empty_cache() 
#    
#output = write_results(pred,.5,80)

#import matplotlib.pyplot as plt
#import cv2
#
#tmp = np.transpose(inp.cpu().numpy().squeeze(),(1,2,0)).copy()
##x1,y1,x2,y2 = output[0,1:5].cpu().numpy()
##cv2.rectangle(tmp,(x1,y1),(x2,y2),(255,0,0),2)
###cv2.rectangle(tmp,(y1,x1),(y2,x2),(255,0,0),2)
##plt.imshow(tmp)
##plt.show()
#
#model.net_info["height"] = 416
#inp_dim = int(model.net_info["height"])
#
##Set the model in evaluation mode
#model.eval()
#loaded_ims = [tmp]
#im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(loaded_ims))]))
#im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
#im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
#batch = im_batches[0].cuda()
#with torch.no_grad():
#    prediction = model(Variable(batch),torch.cuda.is_available())
#
#print(prediction)
#prediction = write_results(prediction, .5, 80)
#output = prediction

#im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
#
#scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)
#
#output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
#output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
#
#output[:,1:5] /= scaling_factor

        
            
        