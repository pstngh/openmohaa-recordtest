"""Learn temporal continuity calibration from validation demonstrations.
GPL-2.0-or-later. No fixed action hold timers or tactical randomness.

A 50-Hz MAP classifier tends to repeat a command forever because changes are rare.
Sample the learned joint policy instead. Four fitted persistence logit offsets
calibrate category-change probabilities without adding random camera noise.
"""
import numpy as np
import torch
from model import unpack
from features import HIDDEN,K,CATS

def previous_classes(x):
    xy=(x[...,27].round().long()+1)*3+x[...,28].round().long()+1
    vertical=x[...,29].round().long()+1
    lean=torch.where(x[...,31]>0,0,torch.where(x[...,32]>0,2,1))
    bits=(x[...,30]>0).long()+2*(x[...,33]>0)+4*(x[...,34]>0)+8*(x[...,35]>0)
    return torch.stack((xy,vertical,lean,bits),-1)

@torch.no_grad()
def logits_for(model,data):
    model.eval();h=torch.zeros(1,48,HIDDEN)
    rows=torch.zeros(len(data.x),144)
    for (x,y,t,m,reset,ay,am),indices in data.batches():
        h[:,reset]=0;out,h,_=model(x,h)
        valid=indices>=0;rows[indices[valid]]=out[valid]
    return rows

def tilted(out,previous,bias):
    pi,cats,means,logs=unpack(out)
    mix=pi; heads=[]
    for j,(cat,n) in enumerate(zip(cats,CATS)):
        tilt=cat+torch.nn.functional.one_hot(previous[...,j],n).float()[...,None,:]*bias[j]
        mix=mix+torch.logsumexp(tilt,-1)
        heads.append(tilt.log_softmax(-1))
    return mix.log_softmax(-1),heads,means

def calibrate(model,data):
    out=logits_for(model,data);previous=previous_classes(torch.from_numpy(data.x));truth=torch.from_numpy(data.y)
    beta=torch.nn.Parameter(torch.zeros(4));opt=torch.optim.LBFGS([beta],max_iter=40,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();mix,heads,_=tilted(out,previous,beta.clamp(-8,8));lp=mix
        for j,head in enumerate(heads):
            lp=lp+head.gather(-1,truth[:,j,None,None].expand(-1,K,1)).squeeze(-1)
        loss=-torch.logsumexp(lp,-1).mean()+.0001*beta.square().sum()
        loss.backward();return loss
    before=float(closure().detach());opt.step(closure);after=float(closure().detach())
    model.stay_bias.copy_(beta.detach().clamp(-8,8))
    return dict(stay_bias=model.stay_bias.tolist(),validation_category_nll_before=before,validation_category_nll_after=after,
                method='Four persistence offsets fitted by validation categorical likelihood; stochastic joint-component/category sampling, no Gaussian camera noise. No hard action-duration timers.')

@torch.no_grad()
def evaluate_calibrated(model,data):
    out=logits_for(model,data);previous=previous_classes(torch.from_numpy(data.x));truth=torch.from_numpy(data.y)
    mix,heads,means=tilted(out,previous,model.stay_bias);weight=mix.exp();measures={}
    for j,head in enumerate(heads):
        probs=(weight[...,None]*head.exp()).sum(-2)
        expected_change=1-probs.gather(-1,previous[:,j,None]).squeeze(-1)
        actual_change=truth[:,j]!=previous[:,j]
        true_prob=probs.gather(-1,truth[:,j,None]).squeeze(-1)
        measures[('xy','vertical','lean','buttons')[j]]=dict(
            recorded_change_rate=float(actual_change.float().mean()),expected_sampled_change_rate=float(expected_change.mean()),
            expected_sampled_accuracy=float(true_prob.mean()),
            expected_correct_probability_at_change=float(true_prob[actual_change].mean()) if actual_change.any() else None)
    turn=torch.from_numpy(data.turn);camera=torch.sinh(means.clamp(-5.19,5.19))
    expected_error=(weight[...,None]*(camera-turn[:,None,:]).abs()).sum(-2)
    previous_turn=torch.sinh(torch.from_numpy(data.x[:,36:38])*5)
    return dict(actions=measures,expected_sampled_camera_mae=expected_error.mean(0).tolist(),
                previous_view_delta_baseline_mae=(previous_turn-turn).abs().mean(0).tolist(),
                scope='Expected one-step sampled behaviour on recorded states. Does not establish transition rates, navigation or pre-aim fidelity in autonomous play.')
