#!/usr/bin/env python3
"""Prepare causal next-command imitation data from the supplied client archive.
GPL-2.0-or-later. No game assets or original telemetry are written to source control.
"""
import argparse,bisect,collections,csv,hashlib,io,json,math,re,zipfile
from pathlib import Path
import numpy as np
from features import *

def metadata(z,name):
    return dict(line.split('=',1) for line in z.read(name).decode('utf-8-sig').splitlines() if '=' in line)

def eligible(r):
    return (int(r['pm_type'])==0 and float(r['health'])>0 and int(r['team']) in (2,3,4)
            and not (int(r['pm_flags']) & (4|32|64|128|256|4096)))

def read_inputs(z,name):
    # Input files are CHANGE logs, not one row per command. Forward-fill by command number.
    values=[]; numbers=[]
    for r in csv.DictReader(io.TextIOWrapper(z.open(name),encoding='utf-8-sig')):
        n=int(r['cmd_number'])
        if numbers and n<=numbers[-1]: raise ValueError(f'{name}: nonmonotonic input commands')
        numbers.append(n)
        values.append((int(r['client_msec']),int(r['cmd_server_msec']),
                       float(r['cmd_pitch']),float(r['cmd_yaw']),
                       int(r['forwardmove']),int(r['rightmove']),int(r['upmove']),int(r['buttons'])&63))
    return numbers,values

def prepare(path,out):
    out.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter(); episodes=[]; file_reports=[]
    observed_commands=collections.Counter(); alignment_errors=[]
    seen=set()
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('_meta.txt'):continue
            m=metadata(z,name)
            if m.get('map')!=MAP:continue
            if (m.get('schema')!='1' or m.get('source')!='client_predicted' or
                int(m.get('map_checksum','0'))!=CHECKSUM or m.get('target_game')!='0'):
                counts['incompatible_metadata']+=1;continue
            sid=m['session_id']; group=sid.split('_maps_')[0]
            base=name[:-9]
            nums, inputs=read_inputs(z,base+'_inputs.csv')
            cur=[]; prev=None; last_mouse=(0.,0.); seen_mouse=None
            file_count=0
            def flush():
                nonlocal cur
                if cur:
                    episodes.append((group,sid,cur));cur=[]
            for r in csv.DictReader(io.TextIOWrapper(z.open(base+'_frames.csv'),encoding='utf-8-sig')):
                counts['frame_rows']+=1
                if r['session_id']!=sid:raise ValueError('session ID mismatch')
                if not eligible(r):
                    counts['excluded_not_living_or_special']+=1;flush();prev=None;last_mouse=(0.,0.);continue
                cmd=int(r['cmd_number']);idx=bisect.bisect_right(nums,cmd)-1
                if idx<0:
                    counts['no_past_input']+=1;flush();prev=None;continue
                u=inputs[idx]
                if u[0]>int(r['client_msec']) or u[1]>int(r['cmd_server_msec']):
                    counts['future_input_mismatch']+=1;flush();prev=None;continue
                own=(int(r['forwardmove']),int(r['rightmove']),int(r['upmove']),int(r['buttons'])&63)
                if own != u[4:]:
                    counts['input_state_mismatch']+=1;flush();prev=None;continue
                if any(x not in (-127,0,127) for x in own[:3]):
                    counts['analog_commands_unsupported']+=1;flush();prev=None;continue
                pos=np.array([float(r['origin_'+a]) for a in 'xyz'])
                vel=np.array([float(r['velocity_'+a]) for a in 'xyz'])
                if not np.isfinite(pos).all() or not np.isfinite(vel).all() or np.max(np.abs(pos))>131072 or np.max(np.abs(vel))>10000:
                    raise ValueError('invalid position/velocity')
                identity=(group,int(r['server_msec']),cmd)
                if identity in seen:
                    counts['duplicate_observation']+=1;flush();prev=None;continue
                seen.add(identity)
                obs=features(r,last_mouse)
                time=int(r['server_msec'])
                counts['eligible_rows']+=1
                if prev is not None:
                    old,old_u,old_obs,old_pos=prev
                    dt=time-int(old['server_msec'])
                    command_dt=int(r['cmd_server_msec'])-int(old['cmd_server_msec'])
                    camera=[angle_delta(float(old['view_'+axis]),float(r['view_'+axis]))*TICK_MS/dt for axis in ('pitch','yaw')] if dt>0 else [0,0]
                    distance=float(np.linalg.norm(pos-old_pos))
                    valid=(10<=dt<=40 and 0<command_dt<=60 and cmd>int(old['cmd_number'])
                           and distance <= max(24,1200*dt/1000)
                           and not (int(r['pm_flags'])&8 and not int(old['pm_flags'])&8)
                           and max(abs(a) for a in camera)<=90)
                    # Compare command-angle deltas with realized view deltas to report recoil/offset mismatch.
                    if valid:
                        err=[abs(angle_delta(old_u[2+j],u[2+j])*TICK_MS/dt-camera[j]) for j in range(2)]
                        alignment_errors.append(err)
                        label=action_classes(*own)
                        cur.append((old_obs,label,camera,dt,old['server_msec'],float(old['target_visible'])))
                        file_count+=1
                        counts['training_pairs']+=1
                        observed_commands[label]+=1
                        last_mouse=tuple(camera)
                        # Current feature must contain only the realized view change that just happened.
                        obs=features(r,last_mouse)
                    else:
                        counts['discontinuity']+=1;flush();last_mouse=(0.,0.);obs=features(r,last_mouse)
                prev=(r,u,obs,pos)
            flush()
            file_reports.append(dict(file=name,group=group,pairs=file_count))
            print(name.split('/')[-1],file_count,flush=True)
    # Entire capture UUIDs are indivisible: multiple files with the same UUID cannot cross splits.
    groups=sorted(set(e[0] for e in episodes),key=lambda s:hashlib.sha256(('mohdm6-v1:'+s).encode()).hexdigest())
    if len(groups)<5:raise ValueError('insufficient independent capture groups for train/validation/test')
    n=max(2,round(len(groups)*.15)); assignments={g:('test' if i<n else 'validation' if i<2*n else 'train') for i,g in enumerate(groups)}
    summary={}
    for split in ('train','validation','test'):
        rows=[]; cat=[]; turns=[]; starts=[0]; episode_groups=[]; episode_sessions=[]; ms=[]
        for g,sid,e in episodes:
            if assignments[g]!=split:continue
            rows.extend(x[0] for x in e);cat.extend(x[1] for x in e);turns.extend(x[2] for x in e);ms.extend(x[3] for x in e)
            starts.append(len(rows));episode_groups.append(g);episode_sessions.append(sid)
        x=np.stack(rows); y=np.array(cat,dtype=np.int64); turn=np.array(turns,dtype=np.float32)
        np.savez_compressed(out/(split+'.npz'),x=x,categories=y,turn=turn,starts=np.array(starts),dt=np.array(ms),groups=np.array(episode_groups),sessions=np.array(episode_sessions))
        summary[split]=dict(pairs=len(rows),episodes=len(starts)-1,groups=sorted(g for g in assignments if assignments[g]==split),minutes=sum(ms)/60000,
                            active_turn_fraction=float(np.mean(np.max(np.abs(turn),axis=1)>.1)))
    report=dict(version=VERSION,map=MAP,checksum=CHECKSUM,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),features=FEATURES,
                counts=dict(counts),splits=summary,files=file_reports,
                command_view_delta_absolute_error_p50_p90_p99=np.percentile(alignment_errors,[50,90,99],axis=0).tolist(),
                alignment='Observation at t predicts the next sampled movement/buttons and realized view-angle delta. Input change logs omit mouse-only changes, so cmd_pitch/yaw are NOT camera labels. cmd_msec is zero and NOT used. Command logs are forward-filled by cmd_number to validate movement/buttons; both client/command timestamps checked. Camera labels include client-predicted view effects; no claim of recovering raw mouse motion. Labels summarize the next 10–40ms, not every unobserved subcommand.',
                omitted=['directional traces: recorder geometry implementation unavailable for exact runtime parity','future states','session clock','team','donor player identity','weapon selection labels'],
                limitations=['Client-predicted observations; server deployment has domain shift.','Inferred target selection is approximated by runtime centroid/FOV/LOS; replay tests do not establish parity.','No BSP assets available for corner annotations.','No duration/inactivity playback filter; valid stationary aiming and short episodes retained.'])
    (out/'dataset-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();prepare(a.archive,a.output)
