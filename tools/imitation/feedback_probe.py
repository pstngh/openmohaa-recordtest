#!/usr/bin/env python3
"""Stress-test v1 recurrent feedback with fixed world observations.
SPDX-License-Identifier: GPL-2.0-or-later.
This is NOT a native physics simulator, gameplay test, or human-fidelity metric.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from model import Policy, decode
from calibration import tilted, previous_classes
from features import HIDDEN


def probe(checkpoint, dataset, decoder='sampled', history='intent', seconds=30, trials=48):
    if seconds < 1 or seconds > 120 or trials < 1 or trials > 256:
        raise ValueError('seconds must be 1..120 and trials 1..256')
    torch.set_num_threads(2); torch.manual_seed(421)
    model=Policy(); model.load_state_dict(torch.load(checkpoint,weights_only=True)); model.eval()
    d=np.load(dataset,allow_pickle=False)
    starts=[int(a) for a,b in zip(d['starts'][:-1],d['starts'][1:]) if b-a>=500][:trials]
    if not starts: raise ValueError('no sufficiently long observation sequences')
    x=torch.from_numpy(d['x'][starts]).clone(); n=len(starts)
    # Immobilized-body stress: position fixed, velocity zero, grounded, no enemy.
    # The map is NOT loaded. Physics, geometry and target acquisition are absent.
    x[:,3:6]=0; x[:,14]=1; x[:,27:38]=0; x[:,46:54]=0
    pitch=torch.rad2deg(torch.atan2(x[:,8],x[:,9])); yaw=torch.rad2deg(torch.atan2(x[:,6],x[:,7]))
    h=torch.zeros(1,n,HIDDEN); down=torch.zeros(n); idle=torch.zeros(n); fire=torch.zeros(n); net=torch.zeros(n)
    ticks=int(seconds*50); ids=torch.arange(n)
    def quantize(angle):
        value=torch.trunc(angle*65536/360)
        return ((value+32768)%65536-32768)*360/65536
    with torch.no_grad():
        for _ in range(ticks):
            out,h,_=model(x[:,None],h); out=out[:,0]
            if decoder=='map':
                categories,camera,_=decode(out)
            else:
                mix,heads,means=tilted(out,previous_classes(x),model.stay_bias)
                k=torch.multinomial(mix.exp(),1).squeeze(-1)
                categories=torch.stack([torch.multinomial(head[ids,k].exp(),1).squeeze(-1) for head in heads],-1)
                camera=torch.sinh(means[ids,k].clamp(-5.19,5.19))
            old_pitch,old_yaw=pitch.clone(),yaw.clone()
            pitch=quantize((pitch+camera[:,0]).clamp(-87.890625,87.890625))
            yaw=quantize((yaw+camera[:,1]+180)%360-180)
            delta_yaw=(yaw-old_yaw+180)%360-180
            xy,vertical,lean,bits=categories.unbind(-1)
            x[:,27]=xy//3-1; x[:,28]=xy%3-1; x[:,29]=vertical-1
            x[:,30]=(bits&1)!=0; x[:,31]=lean==0; x[:,32]=lean==2
            x[:,33]=(bits&2)!=0 if history=='intent' else 0
            x[:,34]=(bits&4)!=0 if history=='intent' else 0
            x[:,35]=(bits&8)!=0 if history=='intent' else 0
            x[:,36]=torch.asinh(pitch-old_pitch)/5; x[:,37]=torch.asinh(delta_yaw)/5
            x[:,8]=pitch.deg2rad().sin(); x[:,9]=pitch.deg2rad().cos()
            x[:,6]=yaw.deg2rad().sin(); x[:,7]=yaw.deg2rad().cos()
            down+=pitch>60; idle+=xy==4; fire+=(bits&2)!=0; net+=delta_yaw
    return dict(trials=n,seconds=seconds,decoder=decoder,history=history,
                source_indices=starts,ending_pitch=pitch.tolist(),net_yaw_revolutions=(net/360).tolist(),
                fraction_downward_over60=float(down.sum()/(n*ticks)),fraction_idle_commands=float(idle.sum()/(n*ticks)),
                primary_intent_fraction=float(fire.sum()/(n*ticks)),
                scope='Fixed position, zero velocity, grounded, no target; predicted controls and quantized view feed back for 50-Hz recurrent inference. No map geometry, physics, native weapon execution or real opponent. This diagnoses feedback instability; it does not establish gameplay or imitation fidelity.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--decoder',choices=['sampled','map'],default='sampled')
    p.add_argument('--history',choices=['intent','executed'],default='intent')
    p.add_argument('--seconds',type=int,default=30);p.add_argument('--trials',type=int,default=48)
    a=p.parse_args()
    if a.output.resolve() in (a.checkpoint.resolve(),a.dataset.resolve()):p.error('output would overwrite an input')
    result=probe(a.checkpoint,a.dataset,a.decoder,a.history,a.seconds,a.trials)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
if __name__=='__main__':main()
