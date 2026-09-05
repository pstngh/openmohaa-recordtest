"""Regression tests for telemetry life segmentation and the binary wire contract.
SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt.
"""
import csv
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import import_recordings as imp


def frame(client, t, x, **changes):
    row = dict(schema=3, session_id='s', session_ms=t, map='dm/test', client_id=client,
               is_bot=0, team=3, alive=1, spectator=0, on_ground=1, on_ladder=0,
               zoomed=0, pm_flags=0, cmd_forward=127, cmd_right=0, cmd_up=0,
               buttons=4, weapon='Thompson', clip_ammo=30, reserve_ammo=90)
    for prefix, values in {'origin':(x,0,1), 'velocity':(20,0,0),
                           'eye':(x,0,83), 'bbox_min':(-16,-16,0), 'bbox_max':(16,16,94)}.items():
        row.update({prefix+'_'+axis:value for axis,value in zip('xyz',values)})
    row.update(view_pitch=0,view_yaw=179,view_roll=0)
    row.update(changes)
    return row


def event(kind, actor, t, target=-1, x=0):
    return dict(session_id='s',session_ms=t,event=kind,actor_id=actor,target_id=target,
                position_x=x,position_y=0,position_z=1,fire_mode=0)


def decode(path):
    data=io.BytesIO(path.read_bytes())
    def read(fmt):
        st=struct.Struct('<'+fmt)
        return st.unpack(data.read(st.size))
    def text():
        return data.read(read('I')[0]).decode('ascii')
    assert data.read(8)==b'OMRPL001'
    name=text()
    checksum,mode,protocol,interval,count=read('5I')
    clips=[]
    for _ in range(count):
        cid=text()
        team,x,y,z,duration,weapons=read('i3fII')
        table=[(text(),*read('Iii')) for _ in range(weapons)]
        nf,na=read('II')
        frames=[imp.FRAME.unpack(data.read(imp.FRAME.size)) for _ in range(nf)]
        actions=[read('II') for _ in range(na)]
        clips.append(dict(id=cid,team=team,spawn=(x,y,z),duration=duration,weapons=table,frames=frames,actions=actions))
    assert not data.read()
    return dict(map=name,checksum=checksum,mode=mode,protocol=protocol,interval=interval,clips=clips)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'source'
        self.output=self.root/'output'
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, frames=None, events=None, checksum=True, folder='', suffix='', map_name='dm/test'):
        folder=self.source/folder
        folder.mkdir(exist_ok=True)
        frames=frames if frames is not None else [frame(0,100,0),frame(0,150,1)]
        events=events if events is not None else [event('spawn',0,100)]
        for name,rows in [('frames',frames),('events',events)]:
            with (folder/f'movement_{name}{suffix}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
        meta=f'[session s]\nschema=3\nsession_id=s\nmap={map_name}\nprotocol=8\nsample_hz=20\ng_gametype=1\n'
        if checksum:
            meta+='sv_mapChecksum=-1\n'
        (folder/f'movement_meta{suffix}.txt').write_text(meta)

    def run_import(self, **kwargs):
        report=imp.import_source(self.source,self.output,min_duration=1,**kwargs)
        return report,decode(self.output/report['libraries'][0]['file'])

    def test_binary_contract(self):
        self.write()
        report,lib=self.run_import()
        self.assertEqual((lib['map'],lib['checksum'],lib['interval']),('dm/test',0xffffffff,50))
        c=lib['clips'][0]
        self.assertEqual([f[0] for f in c['frames']],[0,50])
        self.assertEqual([f[1] for f in c['frames']],[0,1])
        self.assertEqual(c['spawn'],(0,0,1))
        self.assertEqual(c['duration'],100)
        self.assertEqual(c['weapons'],[('Thompson',0,30,90)])

    def test_death_ends_victim_not_killer_and_respawn_pools(self):
        events=[event('spawn',0,100),event('spawn',1,100,x=100),event('death',0,175,target=1),
                event('spawn',0,300),event('reload',0,150)]
        frames=[f for t in [100,150,200,250] for f in [frame(0,t,(t-100)/50),frame(1,t,100+(t-100)/50)]]
        frames += [frame(0,300,0),frame(0,350,1)]
        self.write(frames,events)
        report,lib=self.run_import()
        victim=next(c for c in lib['clips'] if c['spawn'][0]==100)
        self.assertEqual(victim['duration'],75)
        self.assertEqual(len(victim['frames']),2)
        killer=next(c for c in lib['clips'] if len(c['frames'])==4)
        self.assertEqual(killer['actions'],[(50,1)])
        self.assertEqual(killer['duration'],200)
        self.assertEqual(report['accepted'],3)
        self.assertEqual([p['clips'] for p in report['libraries'][0]['pools']],[2,1])

    def test_no_cross_gap_interpolation(self):
        self.write([frame(0,100,0),frame(0,150,1),frame(0,500,5),frame(0,550,6)])
        report,lib=self.run_import()
        self.assertEqual(len(lib['clips'][0]['frames']),2)
        self.assertEqual(lib['clips'][0]['duration'],100)
        self.assertEqual(report['libraries'][0]['end_reasons'],{'sampling_gap_or_teleport':1})

    def test_missing_start_is_rejected(self):
        self.write([frame(0,500,0),frame(0,550,1)])
        with self.assertRaisesRegex(ValueError,'no eligible'):
            self.run_import()

    def test_synthetic_boundary_preserves_original_times(self):
        self.write([frame(0,150,1),frame(0,200,2)])
        report,lib=self.run_import()
        self.assertEqual(report['synthetic_start_clips'],1)
        self.assertEqual([f[0] for f in lib['clips'][0]['frames']],[0,50,100])
        self.assertEqual([f[1] for f in lib['clips'][0]['frames']],[0,1,2])

    def test_bot_identity_comes_from_frames(self):
        self.write([frame(0,100,0,is_bot=1),frame(0,150,1,is_bot=1)])
        with self.assertRaisesRegex(ValueError,'no eligible'):
            self.run_import()
        report,_=self.run_import(include_bots=True)
        self.assertEqual(report['accepted'],1)

    def test_ladder_rejected(self):
        self.write([frame(0,100,0),frame(0,150,1,on_ladder=1)])
        with self.assertRaisesRegex(ValueError,'no eligible'):
            self.run_import()

    def test_unknown_checksum_not_guessed(self):
        self.write(checksum=False)
        with self.assertRaisesRegex(ValueError,'no eligible'):
            self.run_import()

    def test_invalid_numbers_fail_with_line(self):
        self.write([frame(0,100,0),frame(0,150,1,velocity_x='NaN')])
        with self.assertRaisesRegex(ValueError,r'movement_frames.csv:3: invalid velocity'):
            self.run_import()
        self.assertFalse(self.output.exists())

    def test_duplicate_sources_are_deduplicated(self):
        self.write()
        self.write(folder='copy',suffix=' (2)')
        report,_=self.run_import()
        self.assertEqual(report['accepted'],1)
        self.assertEqual(report['rejected']['duplicate_clip'],1)

    def test_zip_and_added_columns(self):
        self.write([frame(0,100,0,extra='ignored'),frame(0,150,1,extra='ignored')],suffix=' (3)')
        path=self.root/'input.zip'
        with zipfile.ZipFile(path,'w') as archive:
            for p in self.source.iterdir():
                archive.write(p,p.name)
        report=imp.import_source(path,self.output,min_duration=1)
        self.assertEqual(report['accepted'],1)

    def test_zero_time_events_and_end_exclusion(self):
        self.write(events=[event('spawn',0,100),event('reload',0,100),event('shot',0,150),event('death',1,175,target=0),event('reload',0,175)])
        _,lib=self.run_import()
        self.assertEqual(lib['clips'][0]['actions'],[(0,1),(50,2)])
        self.assertEqual(lib['clips'][0]['duration'],75)

    def test_spectator_not_gameplay_spawn(self):
        self.write([frame(0,100,0,spectator=1),frame(0,150,1)])
        with self.assertRaisesRegex(ValueError,'no eligible'):
            self.run_import()

    def test_unsafe_map_path_rejected(self):
        self.write(map_name='../other')
        with self.assertRaisesRegex(ValueError,'no eligible'):
            self.run_import()

    def test_pre_spawn_frames_ignored(self):
        self.write([frame(0,50,-1),frame(0,100,0),frame(0,150,1)])
        _,lib=self.run_import()
        self.assertEqual([f[0] for f in lib['clips'][0]['frames']],[0,50])


if __name__=='__main__':
    unittest.main()
