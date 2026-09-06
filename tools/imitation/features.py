"""Causal mohdm6 observation/action contract. GPL-2.0-or-later.
Only current and past player observations are features. Future observations are labels.
No player IDs, session clocks, life index, donor route or future target are inputs.
"""
import math
import numpy as np

VERSION = 1
MAP = 'maps/dm/mohdm6.bsp'
CHECKSUM = 1974169620
TICK_MS = 20
CATS = (9, 3, 3, 16)  # joint XY, vertical, lean, run/fire/secondary/use bits
K = 4
HIDDEN = 64
OUTPUT = K + K * (sum(CATS) + 4)
WEAPON_CLASSES = ('none','smg','sniper','rifle','mg','pistol','grenade','shotgun','other')

def weapon_class(name):
    name = name.lower()
    if not name: return 0
    if 'sniper' in name: return 2
    if name in ('mp40','thompson'): return 1
    if name in ('stg 44','mp44','bar'): return 4
    if 'grenade' in name or 'stielhandgranate' in name: return 6
    if 'shotgun' in name: return 7
    if 'colt' in name or 'p38' in name: return 5
    if any(s in name for s in ('garand','kar 98','kar98','kar 98k')): return 3
    return 8

def angle_delta(a, b):
    return (b-a+180.) % 360.-180.

def action_classes(f,r,u,buttons):
    # Input telemetry uses -127/0/127; analog samples are rejected, not quantized silently.
    xy = (int(f)//127+1)*3 + (int(r)//127+1)
    vertical = int(u)//127+1
    lean = 0 if buttons&16 and not buttons&32 else 2 if buttons&32 and not buttons&16 else 1
    bits = ((buttons>>2)&1) | ((buttons&1)<<1) | ((buttons&2)<<1) | (buttons&8)
    return (xy,vertical,lean,bits)

def features(row, previous_mouse=(0.,0.)):
    xyz = [float(row['origin_'+a]) for a in 'xyz']
    vel = [float(row['velocity_'+a]) for a in 'xyz']
    pitch,yaw = [math.radians(float(row['view_'+a])) for a in ('pitch','yaw')]
    flags, buttons = int(row['pm_flags']), int(row['buttons'])
    f = [xyz[0]/2048, xyz[1]/2048, xyz[2]/512, *(v/400 for v in vel),
         math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
         float(row['lean_angle'])/40, float(row['viewheight'])/96,
         float(bool(flags&1)), float(bool(flags&2)), float(int(row['walking'])!=0),
         float(row['health'])/100, max(0,float(row['clip_ammo']))/50,
         max(0,float(row['ammo']))/300]
    wc=weapon_class(row['weapon_name'])
    f += [float(i==wc) for i in range(len(WEAPON_CLASSES))]
    f += [float(row[a+'move'])/127 for a in ('forward','right','up')]
    f += [float(bool(buttons&b)) for b in (4,16,32,1,2,8)]
    f += [math.asinh(v)/5 for v in previous_mouse]
    for axis in range(2):
        for wavelength in (1024,512):
            p=2*math.pi*xyz[axis]/wavelength
            f += [math.sin(p),math.cos(p)]
    # The telemetry's target is inferred, not an authoritative intention label.
    # A narrow common admission rule excludes distant off-axis direct-hit cases.
    present = (int(row['target_visible'])!=0 and int(row['target_confidence'])>0
               and 0<=float(row['target_angular_error'])<=12 and 1<=float(row['target_distance'])<=4096)
    if present:
        d=float(row['target_distance'])
        f += [1., *(float(row['target_relative_'+a])/d for a in ('forward','right','up')),d/4096]
        f += [float(row['target_velocity_'+a])/400 for a in 'xyz']
    else: f += [0.]*8
    # Extra features explicitly represent delta time and past angular change? Time is fixed at 20ms.
    # Contract size is deliberately checked rather than silently extending it.
    result=np.array(f,dtype=np.float32)
    if result.shape != (54,): raise ValueError(f'feature contract mismatch: {result.shape}')
    if not np.isfinite(result).all(): raise ValueError('nonfinite observation')
    return np.clip(result,-16,16)

FEATURES=54
