import unittest
import torch
import numpy as np
from util import *

class Testutil(unittest.TestCase):
    
    def test_bboxiou(self):
        box = torch.tensor([[1,2,3,4]])
        self.assertEqual(bbox_iou(box,box),torch.tensor([[1]]))
        
    def test_letterboximg(self):
        img = np.random.randn(480,360,3)
        self.assertEqual(letterbox_image(img,(416,416)).shape,
                         (416,416,3))
        
        

if __name__ == '__main__':
    unittest.main()