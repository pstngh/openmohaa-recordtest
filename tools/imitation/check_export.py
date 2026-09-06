#!/usr/bin/env python3
"""Check Python/C++ features, recurrent inference and sampled-action parity.
GPL-2.0-or-later. Requires a checkpoint produced locally by train.py and its export.
The checkpoint is loaded with weights_only=True; no arbitrary model code is loaded.
"""
import argparse,json,struct,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from features import FEATURES,HIDDEN,OUTPUT,K,CATS
from model import Policy
from calibration import previous_classes,tilted
from test_data import row
import features as ft

def choose(out,prior,bias,u):
    mix,heads,means=tilted(out[None,None],prior[None,None],bias)
    def draw(prob,uniform):return min(len(prob)-1,int(np.searchsorted(np.cumsum(prob),uniform,side='right')))
    w=mix.exp()[0,0].numpy();k=draw(w,u[0]);cats=[draw(head.exp()[0,0,k].numpy(),u[j+1]) for j,head in enumerate(heads)]
    xy,up,lean,bits=cats
    buttons=(4 if bits&1 else 0)|(1 if bits&2 else 0)|(2 if bits&4 else 0)|(8 if bits&8 else 0)|(16 if lean==0 else 32 if lean==2 else 0)
    camera=torch.sinh(means[0,0,k].clamp(-5.19,5.19)).numpy()
    return np.array([127*(xy//3-1),127*(xy%3-1),127*(up-1),buttons,*camera,w[k],k],np.float32)

def check(checkpoint,export,data_path,binary,output):
    torch.set_num_threads(2);model=Policy();model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
    d=np.load(data_path);rng=np.random.default_rng(713);indices=[];resets=[]
    for a,b in zip(d['starts'][:-1],d['starts'][1:]):
        if b-a<16:continue
        length=min(128,b-a);indices.extend(range(a,a+length));resets.extend([1]+[0]*(length-1))
        if len(indices)>=1024:break
    x=d['x'][indices];uniform=rng.random((len(x),5),dtype=np.float32)
    prior=previous_classes(torch.from_numpy(x));expected=[];h=torch.zeros(1,1,HIDDEN)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);fixture=root/'fixture.bin';result=root/'result.bin'
        with fixture.open('wb') as f:
            f.write(b'IMCHECK1'+struct.pack('<I',len(x)))
            for i,obs in enumerate(x):
                f.write(struct.pack('<I',resets[i]));f.write(obs.astype('<f4').tobytes())
                f.write(prior[i].numpy().astype('<i4').tobytes());f.write(uniform[i].astype('<f4').tobytes())
                if resets[i]:h.zero_()
                with torch.no_grad():
                    out,h,_=model(torch.from_numpy(obs)[None,None],h)
                    a=choose(out[0,0],prior[i],model.stay_bias,uniform[i])
                    expected.append(np.concatenate((out[0,0].numpy(),h[0,0].numpy(),a)))
        command=subprocess.run([str(binary),str(export),str(fixture),str(result)],check=True,capture_output=True,text=True)
        actual=np.fromfile(result,dtype='<f4').reshape(len(x),OUTPUT+HIDDEN+8);expected=np.stack(expected)
        maximum=np.max(np.abs(actual-expected),axis=0)
        if np.max(maximum[:OUTPUT+HIDDEN])>2e-4:raise AssertionError('recurrent inference parity failed')
        if np.max(maximum[OUTPUT+HIDDEN:])>2e-4:raise AssertionError('action sampling parity failed')
        encoded=root/'features.bin';subprocess.run([str(binary),'--features',str(encoded)],check=True,capture_output=True)
        r=row(origin_x=1024,origin_y=-512,origin_z=256,velocity_x=200,velocity_y=-100,velocity_z=0,view_pitch=10,view_yaw=90,
              lean_angle=20,viewheight=82,pm_flags=1,health=80,clip_ammo=20,ammo=150,forwardmove=127,rightmove=-127,upmove=0,
              buttons=4|1|32,target_visible=1,target_confidence=1,target_angular_error=5,target_distance=100,
              target_relative_forward=100,target_relative_right=0,target_relative_up=10,target_velocity_x=40,target_velocity_y=-20,target_velocity_z=0)
        feature_error=float(np.max(np.abs(np.fromfile(encoded,dtype='<f4')-ft.features(r,(2,-4)))))
        if feature_error>2e-5:raise AssertionError('feature encoder parity failed')
    report=dict(steps=len(x),max_logit_error=float(max(maximum[:OUTPUT])),max_hidden_error=float(max(maximum[OUTPUT:OUTPUT+HIDDEN])),
                max_decoded_action_error=float(max(maximum[OUTPUT+HIDDEN:])),max_feature_error=feature_error,
                native_test_output=command.stdout,scope='Numeric inference and decoder parity only. Runtime perception, physics and corner behaviour require game assets.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for name in ('checkpoint','export','data','binary','output'):p.add_argument(name,type=Path)
    a=p.parse_args();check(a.checkpoint,a.export,a.data,a.binary,a.output)
