
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.models.utils.utils import upsample_bilinear

class MSC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, model, pyramids=[0.5, 0.75,1.0, 1.25, 1.5, 1.75 ],flip=False,combine='max'):
        super(MSC, self).__init__()
        self.scale = model
        self.pyramids = pyramids
        self.flip=flip
        self.combine=combine
        self.debug=True
        print('initialized msc-> pyramids:',pyramids,' flip: ',flip)
        return

    def forward(self, x, debug=False):
        if debug:
            import ipdb;ipdb.set_trace()   
        logits_all = [ ]
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]            
            xp = x.clone() if p==1 else  upsample_bilinear(x.clone(), size=size)
            yp = self.scale(xp)
            yp = yp['out_seg']
            yp =yp if p==1 else upsample_bilinear(yp,x.shape[2:] )
            logits_all=logits_all+[yp]
            if self.flip:
                yp_flip=torch.flip(self.scale(torch.flip(xp.clone(),dims=[3]))['out_seg'], dims=[3])
                yp_flip=yp_flip if p == 1 else upsample_bilinear(yp_flip, x.shape[2:])
                logits_all=logits_all+[yp_flip] 
        # Pixel-wise max
        if self.combine=='max':            
            logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
            if self.debug:
                print('doing max')
                self.debug=False
        else :
            logits_max = torch.mean(torch.stack(logits_all), dim=0)
            if self.debug:
                print('doing mean')
                self.debug=False
      
        if self.training:
            return logits_all+ [logits_max]
        else:
            return logits_max

    def freeze_bn(self):
        self.scale.freeze_bn()

    '''
    def train(self):
        self.scale.train()

    def eval(self):
        self.scale.eval()  

    '''
 
