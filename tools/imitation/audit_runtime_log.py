#!/usr/bin/env python3
"""Summarize feedback-v2 diagnostic lines; never labels a run as human-like.
SPDX-License-Identifier: GPL-2.0-or-later. Uses Python's standard library only.
"""
import argparse
import collections
import json
import math
from pathlib import Path


def audit(lines):
    groups = collections.defaultdict(list)
    malformed = 0
    for line in lines:
        if 'imitation_frame ' not in line:
            continue
        try:
            fields = dict(item.split('=', 1) for item in line.split('imitation_frame ', 1)[1].split())
            obs = [float(v) for v in fields['obs'].split(',')]
            requested = [int(v) for v in fields['requested'].split(',')]
            if len(obs) != 54 or len(requested) != 4 or not all(math.isfinite(v) for v in obs):
                raise ValueError('invalid feature/command vector')
            row = dict(t=int(fields['time']), dt=int(fields['dt']), pitch=float(fields['pitch']),
                       yaw=float(fields['yaw']), requested=requested, sent=int(fields['sent_buttons']),
                       visible=int(fields['visible']), aligned=int(fields['aligned']),
                       permitted=int(fields['permitted']), reset=int(fields['reset']), obs=obs)
            if not 0 < row['dt'] <= 100 or not all(math.isfinite(row[v]) for v in ('pitch','yaw')):
                raise ValueError('invalid tick or angles')
            groups[(int(fields['bot']), int(fields['sequence']), fields['model_id'])].append(row)
        except (ValueError, KeyError):
            malformed += 1
    summaries = []
    for (bot, sequence, model), rows in sorted(groups.items()):
        duration = sum(r['dt'] for r in rows)
        counters = collections.Counter()
        travel = yaw_change = 0.
        previous = None
        for r in rows:
            ms = r['dt']; o = r['obs']
            for key, condition in (
                ('downward_over_60_ms', r['pitch'] > 60),
                ('requested_xy_move_ms', any(r['requested'][:2])),
                ('moving_over_10_units_per_second_ms', math.hypot(o[3],o[4])*400 > 10),
                ('visible_enemy_ms', bool(r['visible'])),
                ('aligned_enemy_ms', bool(r['aligned'])),
                ('requested_fire_ms', bool(r['requested'][3] & 1)),
                ('sent_fire_ms', bool(r['sent'] & 1)),
                ('guard_denied_fire_ms', bool(r['requested'][3]&1) and not r['permitted']),
                ('sent_fire_without_permission_ms', bool(r['sent']&1) and not r['permitted'])):
                if condition: counters[key] += ms
            if previous and not r['reset'] and r['t']-previous['t'] == ms:
                yaw_change += (r['yaw']-previous['yaw']+180)%360-180
                before = previous['obs']
                travel += math.sqrt(sum(((o[i]-before[i])*scale)**2 for i,scale in enumerate((2048,2048,512))))
                if bool(o[33]) != bool(previous['requested'][3] & 1):
                    counters['primary_intent_history_mismatches'] += 1
            previous = r
        summaries.append(dict(bot=bot, sequence=sequence, model_id=model, rows=len(rows),
                              observed_ms=duration, travelled_units=travel,
                              net_yaw_revolutions=yaw_change/360, **dict(counters)))
    return dict(sequences=summaries, malformed_lines=malformed,
                scope='Summarizes observed runtime requests, permissions and motion. Sent fire is an input, not an actual bullet. Missing records are not inferred. Downward aim/turning are measurements, not automatic proof of a bug.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.input.resolve() == args.output.resolve(): parser.error('output must not overwrite log')
    try:
        with args.input.open(encoding='utf-8',errors='replace') as stream: report=audit(stream)
        if not report['sequences']: raise ValueError('no valid feedback-v2 imitation_frame records')
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    except (OSError,ValueError) as exc: parser.exit(2,str(exc)+'\n')
    print(f"Summarized {len(report['sequences'])} sequences; malformed lines: {report['malformed_lines']}")
if __name__ == '__main__': main()
