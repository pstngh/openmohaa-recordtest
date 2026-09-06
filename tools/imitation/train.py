#!/usr/bin/env python3
"""Reproducible CPU behavioural cloning with capture-group holdouts. GPL-2.0-or-later."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from features import FEATURES,HIDDEN,K,CATS,TICK_MS
from model import Policy,objective,decode,export,unpack

class Data:
    def __init__(self,path):
        d=np.load(path)
        self.x=d['x'];self.y=d['categories'];self.turn=d['turn'];self.starts=d['starts'];self.dt=d['dt']
        self.aux=np.zeros((len(self.x),4),np.float32);self.aux_mask=np.zeros(len(self.x),np.float32)
        for a,b in zip(self.starts[:-1],self.starts[1:]):
            if b-a<=25:continue
            cumulative=np.concatenate([np.zeros((1,2)),np.cumsum(self.turn[a:b]*self.dt[a:b,None]/20,axis=0)])
            length=b-a-25
            for j,n in enumerate((10,25)):
                self.aux[a:a+length,j*2:j*2+2]=np.arcsinh(cumulative[n:n+length]-cumulative[:length])
            self.aux_mask[a:a+length]=1
    def batches(self,batch=48,steps=64,seed=None):
        order=np.arange(len(self.starts)-1)
        if seed is not None:np.random.default_rng(seed).shuffle(order)
        slots=[None]*batch;cursor=0
        while True:
            x=np.zeros((batch,steps,FEATURES),np.float32);y=np.zeros((batch,steps,4),np.int64)
            turn=np.zeros((batch,steps,2),np.float32);mask=np.zeros((batch,steps),np.float32)
            ay=np.zeros((batch,steps,4),np.float32);am=np.zeros((batch,steps),np.float32)
            reset=np.zeros(batch,np.bool_);indices=np.full((batch,steps),-1,np.int64)
            for i in range(batch):
                if slots[i] is None and cursor<len(order):
                    e=order[cursor];cursor+=1;slots[i]=[self.starts[e],self.starts[e+1]];reset[i]=True
                if slots[i] is None:continue
                a,b=slots[i];n=min(steps,b-a)
                x[i,:n]=self.x[a:a+n];y[i,:n]=self.y[a:a+n];turn[i,:n]=self.turn[a:a+n]
                mask[i,:n]=1;ay[i,:n]=self.aux[a:a+n];am[i,:n]=self.aux_mask[a:a+n]
                indices[i,:n]=np.arange(a,a+n)
                slots[i]=None if a+n==b else [a+n,b]
            if not mask.any():return
            yield tuple(torch.from_numpy(a) for a in (x,y,turn,mask,reset,ay,am)),indices

@torch.no_grad()
def evaluate(model,data,save=None):
    model.eval();h=torch.zeros(1,48,HIDDEN)
    predictions=np.zeros_like(data.y);turns=np.zeros_like(data.turn);nll=0.;count=0;components=np.zeros(K,int)
    for (x,y,t,m,reset,ay,am),indices in data.batches():
        h[:,reset]=0
        out,h,aux=model(x,h)
        pi,cats,means,logs=unpack(out)
        lp=pi
        for c,target in zip(cats,y.unbind(-1)):
            lp=lp+c.gather(-1,target[...,None,None].expand(*target.shape,K,1)).squeeze(-1)
        lp=lp-(((torch.asinh(t)[...,None,:]-means)**2*torch.exp(-2*logs)*.5)+logs+.5*math.log(2*math.pi)).sum(-1)
        losses=-torch.logsumexp(lp,-1)
        nll+=float((losses*m).sum());count+=int(m.sum())
        a,c,k=decode(out);valid=indices>=0
        predictions[indices[valid]]=a.numpy()[valid];turns[indices[valid]]=c.numpy()[valid]
        components+=np.bincount(k.numpy()[valid],minlength=K)
    active=np.max(np.abs(data.turn),axis=1)>.1
    moving=np.linalg.norm(data.x[:,3:5],axis=1)>.1
    no_enemy=data.x[:,46]==0
    # Previous commands are causal persistence baselines, not a second trained policy.
    prevxy=np.rint((data.x[:,27]*127)/127).astype(int)+1
    prevr=np.rint((data.x[:,28]*127)/127).astype(int)+1
    previous=np.stack((prevxy*3+prevr,np.rint(data.x[:,29]).astype(int)+1,
                       np.where(data.x[:,31]>0,0,np.where(data.x[:,32]>0,2,1)),
                       (data.x[:,30]>0).astype(int)+2*(data.x[:,33]>0)+4*(data.x[:,34]>0)+8*(data.x[:,35]>0)),-1)
    change=previous[:,0]!=data.y[:,0]
    meanabs=lambda a,mask: np.mean(a[mask],axis=0).tolist() if np.any(mask) else None
    result=dict(pairs=count,joint_nll=nll/count,
        categorical_accuracy=dict(zip(('xy','vertical','lean','buttons'),np.mean(predictions==data.y,axis=0).tolist())),
        persistence_accuracy=dict(zip(('xy','vertical','lean','buttons'),np.mean(previous==data.y,axis=0).tolist())),
        xy_change_events=int(change.sum()),xy_change_accuracy=float(np.mean(predictions[change,0]==data.y[change,0])) if change.any() else None,
        turn_mae_deg_per_20ms=meanabs(np.abs(turns-data.turn),np.ones(count,bool)),
        zero_turn_mae_deg_per_20ms=meanabs(np.abs(data.turn),np.ones(count,bool)),
        active_turn_count=int(active.sum()),active_turn_mae=meanabs(np.abs(turns-data.turn),active),
        active_turn_zero_baseline_mae=meanabs(np.abs(data.turn),active),
        moving_without_selected_enemy_count=int((active&moving&no_enemy).sum()),
        moving_without_selected_enemy_turn_mae=meanabs(np.abs(turns-data.turn),active&moving&no_enemy),
        selected_components=components.tolist(),
        evaluation='Teacher-forced next-step prediction on entire held-out capture groups. Not autonomous rollout or per-corner fidelity validation.')
    if save is not None:np.savez_compressed(save,actions=predictions,turn=turns)
    return result

def train(data_dir,out,epochs=18):
    torch.manual_seed(1729);np.random.seed(1729);random.seed(1729);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)
    train=Data(data_dir/'train.npz');val=Data(data_dir/'validation.npz')
    model=Policy();model.mean.copy_(torch.from_numpy(train.x.mean(0)));model.std.copy_(torch.from_numpy(np.maximum(train.x.std(0),.05)))
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
    best=float('inf');history=[];start=time.time()
    for epoch in range(1,epochs+1):
        model.train();h=torch.zeros(1,48,HIDDEN);loss_total=0;num=0
        for (x,y,t,m,reset,ay,am),indices in train.batches(seed=1729+epoch):
            h=h.detach();h[:,reset]=0
            outv,h,aux=model(x,h);loss=objective(outv,y,t,m,aux,ay,am)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();loss_total+=float(loss.detach())*int(m.sum());num+=int(m.sum())
        metrics=evaluate(model,val)
        score=metrics['joint_nll'];improved=score<best
        if improved:
            best=score;torch.save(model.state_dict(),out/'best.pt');best_epoch=epoch
            export(model,out/'mohdm6.omim')
        row=dict(epoch=epoch,train_loss=loss_total/num,validation=metrics,seconds=round(time.time()-start,1),selected=improved)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps({'epoch':epoch,'train_loss':row['train_loss'],'val_nll':score,'val_xy':metrics['categorical_accuracy']['xy'],'val_turn':metrics['active_turn_mae'],'best':improved,'seconds':row['seconds']}),flush=True)
        if epoch%6==0:
            for p in opt.param_groups:p['lr']*=.7
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    from calibration import calibrate,evaluate_calibrated
    calibration=calibrate(model,val)
    export(model,out/'mohdm6.omim')
    torch.save(model.state_dict(),out/'best-calibrated.pt')
    test=evaluate(model,Data(data_dir/'test.npz'),out/'test-predictions.npz')
    report=dict(model='Joint four-component categorical/Gaussian GRU policy',seed=1729,torch=torch.__version__,epochs=epochs,best_epoch=best_epoch,
                hidden=HIDDEN,features=FEATURES,parameters=sum(p.numel() for p in model.parameters()),
                validation=evaluate(model,val),test=test,calibration=calibration,
                calibrated_validation=evaluate_calibrated(model,val),calibrated_test=evaluate_calibrated(model,Data(data_dir/'test.npz')),model_sha256=hashlib.sha256((out/'mohdm6.omim').read_bytes()).hexdigest(),
                map='dm/mohdm6',checksum=1974169620,training_source='client_telemetry.zip, mohdm6 only',
                omitted='No reward optimization, tactical routes, other players/maps, playback clock, positional overrides, or live learning.',
                status='Offline-trained experimental checkpoint. Not gameplay-validated; no promise of natural autonomous trajectories.')
    (out/'training-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--epochs',type=int,default=18);a=p.parse_args();train(a.data,a.output,a.epochs)
