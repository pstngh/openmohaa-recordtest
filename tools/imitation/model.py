"""Joint recurrent mixture policy, exported to dependency-free C++ inference.
GPL-2.0-or-later. Trained only on user-supplied mohdm6 demonstrations.
"""
import math,struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from features import FEATURES,HIDDEN,OUTPUT,K,CATS,CHECKSUM,TICK_MS

COMPONENT=sum(CATS)+4
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(FEATURES))
        self.register_buffer('std',torch.ones(FEATURES))
        self.register_buffer('stay_bias',torch.zeros(len(CATS)))
        self.gru=nn.GRU(FEATURES,HIDDEN,batch_first=True)
        self.head=nn.Linear(HIDDEN,OUTPUT)
        self.future=nn.Linear(HIDDEN,4) # auxiliary future camera change; never a runtime feature
    def forward(self,x,h=None):
        x=((x-self.mean)/self.std).clamp(-10,10)
        z,h=self.gru(x,h)
        return self.head(z),h,self.future(z)

def unpack(out):
    pi=out[...,:K].log_softmax(-1)
    component=out[...,K:].reshape(*out.shape[:-1],K,COMPONENT)
    categories=[]; off=0
    for n in CATS:
        categories.append(component[...,off:off+n].log_softmax(-1));off+=n
    return pi,categories,component[...,off:off+2],component[...,off+2:].clamp(-3,1.5)

def objective(out,y,turn,mask,aux,aux_y,aux_mask):
    pi,cats,means,logstd=unpack(out)
    lp=pi
    for head,target in zip(cats,y.unbind(-1)):
        lp=lp+head.gather(-1,target[...,None,None].expand(*target.shape,K,1)).squeeze(-1)
    z=torch.asinh(turn)[...,None,:]
    lp=lp-((z-means).square()*torch.exp(-2*logstd)*.5+logstd+.5*math.log(2*math.pi)).sum(-1)
    loss=-torch.logsumexp(lp,-1)
    # Emphasize actual demonstrated changes without inventing control targets.
    weights=1.+(turn.abs().amax(-1)>.1).float()
    result=(loss*weights*mask).sum()/(weights*mask).sum().clamp_min(1)
    aux_loss=nn.functional.smooth_l1_loss(aux,aux_y,reduction='none').mean(-1)
    return result+.05*(aux_loss*aux_mask*mask).sum()/(aux_mask*mask).sum().clamp_min(1)

def decode(out):
    pi,cats,means,logs=unpack(out)
    component=pi.argmax(-1)
    action=torch.stack([h.gather(-2,component[...,None,None].expand(*component.shape,1,n)).squeeze(-2).argmax(-1) for h,n in zip(cats,CATS)],-1)
    camera=torch.sinh(means.gather(-2,component[...,None,None].expand(*component.shape,1,2)).squeeze(-2).clamp(-5.19,5.19))
    return action,camera,component

def export(model,path):
    names=['mean','std','gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','head.weight','head.bias','stay_bias']
    data=model.state_dict()
    floats=np.concatenate([data[n].detach().cpu().numpy().reshape(-1) for n in names]).astype('<f4')
    raw=b'OMIM0001'+struct.pack('<7I',FEATURES,HIDDEN,K,OUTPUT,CHECKSUM,TICK_MS,len(floats))+floats.tobytes()
    Path(path).write_bytes(raw)
