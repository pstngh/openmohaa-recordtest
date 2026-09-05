#!/usr/bin/env python3
"""Import spawn-anchored lives from movement telemetry into OMRPL001 libraries.

Copyright (C) 2026 OpenMoHAA contributors
SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt.
Uses Python's standard library only. Never executes recording text as commands.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import configparser
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field

FRAME = struct.Struct('<I18fI3i3If')
assert FRAME.size == 108
MAX_FILE = 128 * 1024 * 1024
MAX_FRAMES = 1_000_000


def integer(row: dict, key: str, default=None) -> int:
    text = row.get(key, default)
    if text is None or text == '':
        raise ValueError(f'missing {key}')
    return int(text)


def vector(row: dict, prefix: str, suffixes=('x', 'y', 'z'), limit=131072) -> tuple:
    values = tuple(float(row[prefix + '_' + suffix]) for suffix in suffixes)
    if any(not math.isfinite(v) or abs(v) > limit for v in values):
        raise ValueError(f'invalid {prefix}')
    return values


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def safe_map(name):
    return bool(re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', name))


class Source:
    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.is_file() else None
        self.names = sorted(self.zip.namelist() if self.zip else
                            (p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()))

    def open(self, name):
        if self.zip:
            return io.TextIOWrapper(self.zip.open(name), encoding='utf-8-sig', errors='strict', newline='')
        return (self.path / name).open(encoding='utf-8-sig', newline='')

    def close(self):
        if self.zip:
            self.zip.close()

    def sets(self):
        for name in self.names:
            path = PurePosixPath(name)
            match = re.fullmatch(r'movement_frames(.*)\.csv', path.name)
            if not match:
                continue
            suffix = match.group(1)
            events = str(path.with_name(f'movement_events{suffix}.csv'))
            meta = str(path.with_name(f'movement_meta{suffix}.txt'))
            if events not in self.names or meta not in self.names:
                raise ValueError(f'missing matching event/metadata file for {name}')
            yield name, events, meta


@dataclass
class Life:
    session: str
    client: int
    start: int
    spawn: tuple
    end: int = 2**31-1
    reason: str = 'recording_end'
    team: int | None = None
    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    weapons: list = field(default_factory=list)
    weapon_ids: dict = field(default_factory=dict)
    stopped: bool = False
    rejected: str = ''
    synthetic_start: bool = False


def read_sessions(source, name):
    config = configparser.ConfigParser(interpolation=None, strict=True)
    with source.open(name) as stream:
        config.read_file(stream)
    return {section.get('session_id', title.removeprefix('session ')): dict(section)
            for title in config.sections() for section in [config[title]]}


def read_events(source, name):
    lives = collections.defaultdict(list)
    boundaries = collections.defaultdict(list)
    actions = collections.defaultdict(list)
    session_ends = {}
    with source.open(name) as stream:
        for row in csv.DictReader(stream):
            sid = row['session_id']
            t = integer(row, 'session_ms')
            kind = row['event']
            if kind == 'session_end':
                session_ends[sid] = t
            elif kind == 'spawn':
                client = integer(row, 'actor_id')
                lives[sid, client].append(Life(sid, client, t, vector(row, 'position')))
            elif kind in ('death', 'client_disconnect'):
                # A death's actor is the killer; its TARGET is the deceased player.
                client = integer(row, 'target_id' if kind == 'death' else 'actor_id')
                boundaries[sid, client].append((t, kind))
            elif kind in ('reload', 'shot'):
                action = 1 if kind == 'reload' else (3 if integer(row, 'fire_mode') == 1 else 2)
                actions[sid, integer(row, 'actor_id')].append((t, action))
    starts = {}
    for key, pool in lives.items():
        pool.sort(key=lambda life: life.start)
        if len({life.start for life in pool}) != len(pool):
            raise ValueError(f'duplicate spawn timestamp in {name}: {key}')
        starts[key] = [life.start for life in pool]
        stops = sorted(boundaries[key])
        stop_times = [event[0] for event in stops]
        for i, life in enumerate(pool):
            if i+1 < len(pool):
                life.end, life.reason = pool[i+1].start, 'next_spawn'
            j = bisect.bisect_left(stop_times, life.start)
            if j < len(stops) and stops[j][0] < life.end:
                life.end, life.reason = stops[j]
            if session_ends.get(key[0], 2**31-1) < life.end:
                life.end, life.reason = session_ends[key[0]], 'session_end'
        for t, action in sorted(actions[key]):
            i = bisect.bisect_right(starts[key], t)-1
            if i >= 0 and t < pool[i].end:
                pool[i].actions.append((t-pool[i].start, action))
    return lives, starts


def lean_step(angle, buttons, msec):
    # Approximation: these CSV schemas store lean inputs, not authoritative lean angles.
    target = -40.0 if buttons & 16 and not buttons & 32 else (40.0 if buttons & 32 and not buttons & 16 else 0.0)
    return target + (angle-target) * math.exp(-(10 if target else 15)*msec/1000)


def add_frame(life: Life, row, sample_ms: int, include_bots: bool):
    if life.stopped:
        return
    t = integer(row, 'session_ms')-life.start
    if integer(row, 'spectator') or not integer(row, 'alive'):
        life.end = min(life.end, t+life.start)
        life.reason, life.stopped = 'nonliving_or_spectator', True
        return
    if integer(row, 'is_bot') and not include_bots:
        life.rejected, life.stopped = 'bot_recording', True
        return
    if integer(row, 'on_ladder'):
        life.rejected, life.stopped = 'unsupported_ladder', True
        return
    team = integer(row, 'team')
    if team < 2 or team > 4:
        life.rejected, life.stopped = 'invalid_team', True
        return
    if life.team is not None and life.team != team:
        life.reason, life.stopped = 'team_change', True
        return
    life.team = team
    origin = vector(row, 'origin')
    velocity = vector(row, 'velocity', limit=10000)
    angles = vector(row, 'view', ('pitch', 'yaw', 'roll'), 3600)
    mins, maxs = vector(row, 'bbox_min', limit=256), vector(row, 'bbox_max', limit=256)
    if any(a >= b for a, b in zip(mins, maxs)):
        raise ValueError('invalid bounding box')
    eye = tuple(a-b for a,b in zip(vector(row,'eye'), origin))
    if any(abs(x)>256 for x in eye):
        raise ValueError('invalid eye offset')
    if not life.frames:
        if t > sample_ms*2 or distance(origin, life.spawn) > 32:
            life.rejected, life.stopped = 'missing_spawn_samples', True
            return
        if t == 0 and distance(origin, life.spawn) > .1:
            life.rejected, life.stopped = 'ambiguous_spawn_sample', True
            return
    else:
        dt = t-life.frames[-1][0]
        if dt <= 0:
            raise ValueError('nonmonotonic or duplicate player frame')
        if dt > sample_ms*2 or distance(origin, life.frames[-1][1:4]) > max(64, 2000*dt/1000):
            life.reason, life.stopped = 'sampling_gap_or_teleport', True
            return
    life.team = team
    pm_flags = integer(row,'pm_flags')
    commands = tuple(integer(row,'cmd_'+axis) for axis in ('forward','right','up'))
    if not 0 <= pm_flags <= 0xffffffff or any(c < -127 or c > 127 for c in commands):
        raise ValueError('invalid movement command')
    buttons = integer(row, 'buttons') & 63
    pose = integer(row,'on_ground') | (integer(row,'zoomed') << 2)
    name = row['weapon']
    if not name.isascii() or len(name)>64 or any(ord(c)<32 for c in name):
        raise ValueError('invalid weapon name')
    if name not in life.weapon_ids:
        if len(life.weapons) >= 64:
            raise ValueError('too many weapons')
        ammo, reserve = integer(row,'clip_ammo'), integer(row,'reserve_ammo')
        if not -1 <= ammo <= 10000 or not -1 <= reserve <= 100000:
            raise ValueError('invalid ammo')
        life.weapon_ids[name] = len(life.weapons)
        life.weapons.append((name,t,ammo,reserve))
    weapon = life.weapon_ids[name]
    lean = lean_step(life.frames[-1][-1] if life.frames else 0, buttons,
                     t-life.frames[-1][0] if life.frames else 0)
    frame = (t,*origin,*velocity,*angles,*mins,*maxs,*eye,pm_flags,*commands,buttons,pose,weapon,lean)
    if not life.frames and t > 0:
        # A visibly marked interpolation boundary, not a fabricated original observation.
        synthetic = (0,*life.spawn,0.,0.,0.,*angles,*mins,*maxs,*eye,pm_flags,0,0,0,0,pose,weapon,0.)
        life.frames.append(synthetic)
        life.synthetic_start = True
        life.weapons[weapon] = (name,0,*life.weapons[weapon][2:])
    life.frames.append(frame)


def text(stream, value):
    raw = value.encode('ascii')
    stream.write(struct.pack('<I',len(raw)))
    stream.write(raw)


def encode_library(key, clips, sample_ms):
    map_name, checksum, game_type, protocol = key
    out = io.BytesIO()
    out.write(b'OMRPL001')
    text(out,map_name)
    out.write(struct.pack('<5I',checksum,game_type,protocol,sample_ms,len(clips)))
    if sum(len(life.frames) for life,_,_ in clips)>MAX_FRAMES:
        raise ValueError('library exceeds one million frames; select a smaller source subset')
    for life,duration,clip_id in clips:
        text(out,clip_id)
        out.write(struct.pack('<i3fII',life.team,*life.spawn,duration,len(life.weapons)))
        for name,t,ammo,reserve in life.weapons:
            text(out,name)
            out.write(struct.pack('<Iii',t,ammo,reserve))
        actions = [(t,k) for t,k in life.actions if t<duration]
        out.write(struct.pack('<II',len(life.frames),len(actions)))
        for frame in life.frames:
            out.write(FRAME.pack(*frame))
        for action in actions:
            out.write(struct.pack('<II',*action))
        if out.tell()>MAX_FILE:
            raise ValueError('library exceeds 128 MiB; select a smaller source subset')
    return out.getvalue()


def import_source(path: Path, output: Path, map_filter=None, include_bots=False, min_duration=500):
    source = Source(path)
    groups = collections.defaultdict(list)
    intervals = {}
    report = {'format':'OMRPL001','accepted':0,'rejected':{},'libraries':[],
              'synthetic_start_clips':0,'note':'Positions are float32 CSV samples; lean and between-sample motion are reconstructed.'}
    rejected = collections.Counter()
    seen = set()
    try:
        sets = list(source.sets())
        if not sets:
            raise ValueError('no matching movement_frames/events/meta file sets found')
        for frames_name, events_name, meta_name in sets:
            sessions = read_sessions(source,meta_name)
            lives, starts = read_events(source,events_name)
            session_keys = {}
            for sid,meta in sessions.items():
                name = meta.get('map','')
                if map_filter and name != map_filter:
                    continue
                if not safe_map(name) or meta.get('protocol')!='8' or 'sv_mapchecksum' not in meta:
                    continue # Unknown BSP identity is never guessed for older schemas.
                if int(meta.get('schema','0')) not in (2,3,4,6):
                    raise ValueError(f'unsupported schema in {meta_name}: {sid}')
                key = name,int(meta['sv_mapchecksum']) & 0xffffffff,int(meta['g_gametype']),int(meta['protocol'])
                hz = float(meta['sample_hz'])
                if not math.isfinite(hz) or hz<1 or hz>1000 or abs(1000/hz-round(1000/hz))>.0001:
                    raise ValueError('sample_hz must have an integral millisecond interval')
                sample_ms = round(1000/hz)
                if key in intervals and intervals[key]!=sample_ms:
                    raise ValueError('inconsistent sample interval for the same map/checksum/mode')
                intervals[key] = sample_ms
                session_keys[sid] = key
            with source.open(frames_name) as stream:
                for line,row in enumerate(csv.DictReader(stream),2):
                    sid = row.get('session_id','')
                    if sid not in session_keys:
                        continue
                    key = sid,integer(row,'client_id')
                    if key not in lives:
                        continue
                    t = integer(row,'session_ms')
                    i = bisect.bisect_right(starts[key],t)-1
                    if i<0 or t>=lives[key][i].end:
                        continue
                    life = lives[key][i]
                    try:
                        if row['map'] != session_keys[sid][0]:
                            raise ValueError('map changed within session')
                        add_frame(life,row,intervals[session_keys[sid]],include_bots)
                    except (ValueError,KeyError,OverflowError) as exc:
                        raise ValueError(f'{frames_name}:{line}: {exc}') from exc
            for pool in lives.values():
                for life in pool:
                    key = session_keys.get(life.session)
                    if key is None:
                        rejected['filtered_or_unknown_map_checksum']+=1
                        continue
                    duration = min(life.end-life.start,
                                   life.frames[-1][0]+intervals[key] if life.frames else 0)
                    reason = life.rejected or ('short_or_missing_life' if len(life.frames)<2 or duration<min_duration else '')
                    if not reason and (duration<=0 or duration>3600000):
                        reason='invalid_duration'
                    if reason:
                        rejected[reason]+=1
                        continue
                    digest=hashlib.sha256()
                    digest.update(f'{key}:{life.session}:{life.client}:{life.start}'.encode())
                    for frame in life.frames:
                        digest.update(FRAME.pack(*frame))
                    clip_id=digest.hexdigest()[:32]
                    if clip_id in seen:
                        rejected['duplicate_clip']+=1
                        continue
                    seen.add(clip_id)
                    groups[key].append((life,duration,clip_id))
                    report['synthetic_start_clips']+=int(life.synthetic_start)
    finally:
        source.close()
    if not groups:
        raise ValueError('no eligible lives (matching checksum, protocol 8, living frames and spawn needed)')
    for key,clips in sorted(groups.items()):
        clips.sort(key=lambda clip:clip[2])
        data=encode_library(key,clips,intervals[key])
        relative=f'{key[0]}.{key[1]}.{key[2]}.rpl'
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(data)
        temporary.replace(destination)
        pools=collections.Counter((life.team if key[2]!=1 else 0,*life.spawn) for life,_,_ in clips)
        report['libraries'].append({'file':relative,'map':key[0],'checksum':key[1],'game_type':key[2],
                                   'clips':len(clips),'frames':sum(len(life.frames) for life,_,_ in clips),
                                   'sha256':hashlib.sha256(data).hexdigest(),
                                   'pools':[{'team':p[0],'spawn':p[1:],'clips':n} for p,n in sorted(pools.items())],
                                   'end_reasons':dict(collections.Counter(life.reason for life,_,_ in clips))})
        report['accepted']+=len(clips)
    report['rejected']=dict(sorted(rejected.items()))
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='ZIP or directory containing paired telemetry files')
    parser.add_argument('output',type=Path,help='game replays directory, e.g. /games/mohaa/main/replays')
    parser.add_argument('--map',dest='map_filter',help='exact map name, e.g. dm/crnodoors')
    parser.add_argument('--include-bots',action='store_true',help='also import bot lives; default is humans only')
    parser.add_argument('--min-duration-ms',type=int,default=500)
    args=parser.parse_args()
    try:
        if args.min_duration_ms<1:
            raise ValueError('minimum duration must be positive')
        report=import_source(args.input,args.output,args.map_filter,args.include_bots,args.min_duration_ms)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,configparser.Error) as exc:
        parser.exit(2,f'import failed: {exc}\n')
    print(f"Imported {report['accepted']} lives in {len(report['libraries'])} libraries; see {args.output/'manifest.json'}")


if __name__=='__main__':
    main()
