"""Regression tests for telemetry life segmentation and the binary wire contract.
SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt.
"""
import csv
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
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
        # Structural/segmentation fixtures are intentionally tiny. Quality tests
        # below exercise production defaults independently.
        settings = dict(min_duration=1, max_stationary_ms=0, max_stationary_fraction=1)
        settings.update(kwargs)
        report=imp.import_source(self.source,self.output,**settings)
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
        report=imp.import_source(path,self.output,min_duration=1,max_stationary_ms=0,max_stationary_fraction=1)
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


class ActivityTests(unittest.TestCase):
    def metrics(self, points, duration, speed=10):
        # The analyzer deliberately reads only timestamp and XY, not velocity or aim.
        return imp.activity_metrics(points, duration, speed)

    def test_motion_is_horizontal_position_not_buttons_velocity_or_aim(self):
        points = [(0, 0, 0, 1, 200, 0, 0), (500, 0, 0, 60, 200, 0, 180),
                  (1000, 0, 0, 1, 200, 0, 359)]
        result = self.metrics(points, 1500)
        self.assertEqual(result['stationary_ms'], 1500)
        self.assertEqual(result['longest_stationary_ms'], 1500)
        self.assertEqual(result['stationary_fraction'], 1)
        self.assertEqual(result['distance_xy'], 0)

    def test_time_weighting_and_terminal_hold(self):
        result = self.metrics([(0, 0, 0), (100, 10, 0), (300, 10, 0),
                               (350, 20, 0), (400, 30, 0)], 500)
        self.assertEqual(result['stationary_ms'], 300)
        self.assertEqual(result['longest_stationary_ms'], 200)
        self.assertEqual(result['stationary_fraction'], .60)
        self.assertEqual(result['distance_xy'], 30)

    def test_tiny_horizontal_noise_does_not_reset_stationary_timer(self):
        points = [(t, 0.1 if t % 100 else 0, 0) for t in range(0, 500, 50)]
        self.assertEqual(self.metrics(points, 500)['longest_stationary_ms'], 500)

    def test_speed_threshold_is_inclusive_and_configurable(self):
        points = [(0, 0, 0), (100, 1, 0), (200, 2, 0)]
        self.assertEqual(self.metrics(points, 300, 10)['stationary_ms'], 300)
        self.assertEqual(self.metrics(points, 300, 9)['stationary_ms'], 100)

    def test_y_axis_strafing_counts_as_motion(self):
        points = [(0, 0, 0), (100, 0, 10), (200, 0, 0)]
        result = self.metrics(points, 300)
        self.assertEqual(result['stationary_ms'], 100)
        self.assertEqual(result['distance_xy'], 20)

    def test_final_hold_extends_stationary_run(self):
        result = self.metrics([(0, 0, 0), (100, 10, 0), (200, 10, 0)], 500)
        self.assertEqual(result['longest_stationary_ms'], 400)

    def test_resampling_constant_speed_does_not_change_time_metrics(self):
        coarse = [(0, 0, 0), (100, 20, 0), (300, 20, 0), (400, 40, 0)]
        fine = [(0, 0, 0), (50, 10, 0), (100, 20, 0), (200, 20, 0),
                (300, 20, 0), (350, 30, 0), (400, 40, 0)]
        self.assertEqual(self.metrics(coarse, 500), self.metrics(fine, 500))

    def test_threshold_boundaries_and_primary_reason_order(self):
        filters = imp.QualityFilters()
        metrics = dict(duration_ms=10000, longest_stationary_ms=3000, stationary_fraction=.60)
        self.assertEqual(imp.quality_reasons(metrics, filters), [])
        metrics.update(duration_ms=9999, longest_stationary_ms=3001, stationary_fraction=.6001)
        self.assertEqual(imp.quality_reasons(metrics, filters),
                         ['life_too_short', 'stationary_stretch_too_long', 'too_much_stationary_time'])

    def test_nonfinite_or_invalid_settings_are_rejected_before_io(self):
        for changes in [dict(min_duration=0), dict(min_duration=1.5), dict(min_duration=True),
                        dict(max_stationary_ms=-1), dict(max_stationary_ms=1.5),
                        dict(max_stationary_fraction=-.1), dict(max_stationary_fraction=1.1),
                        dict(max_stationary_fraction=float('nan')), dict(max_stationary_fraction=float('inf')),
                        dict(stationary_speed=-1), dict(stationary_speed=float('inf')),
                        dict(stationary_speed=float('nan'))]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                imp.import_source(Path('/nonexistent-recording-input'), Path('/unused-output'), **changes)


class QualityImportTests(unittest.TestCase):
    # Reuse file-writing helpers without inheriting/rerunning the structural tests.
    setUp = ImportTests.setUp
    tearDown = ImportTests.tearDown
    write = ImportTests.write

    def moving(self, client=0, duration=10000, spawn=0):
        return [frame(client, t, spawn + t / 50) for t in range(0, duration, 50)]

    def test_production_defaults_reject_short_life_accept_exact_ten_seconds(self):
        frames = self.moving(0, 9950) + self.moving(1, 10000)
        self.write(frames, [event('spawn', 0, 0), event('spawn', 1, 0)])
        report = imp.import_source(self.source, self.output)
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['rejected'], {'life_too_short': 1})
        self.assertEqual(report['quality']['candidates'], 2)
        self.assertEqual(report['quality']['settings']['min_duration_ms'], 10000)
        self.assertEqual(report['quality']['pools'][0]['accepted'], 1)

    def test_initial_stationary_stretch_rejected_without_cutting_or_reanchoring(self):
        frames = self.moving(0) + [frame(1, t, max(0, t - 3050) / 50) for t in range(0, 10000, 50)]
        self.write(frames, [event('spawn', 0, 0), event('spawn', 1, 0)])
        report = imp.import_source(self.source, self.output)
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['rejected'], {'stationary_stretch_too_long': 1})
        rejected = next(c for c in report['quality']['clips'] if not c['accepted'])
        self.assertEqual(rejected['longest_stationary_ms'], 3050)
        self.assertEqual(rejected['spawn'], (0, 0, 1))

    def test_exactly_three_seconds_stationary_is_allowed(self):
        self.write([frame(0, t, max(0, t - 3000) / 50) for t in range(0, 10000, 50)],
                   [event('spawn', 0, 0)])
        report = imp.import_source(self.source, self.output)
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['quality']['clips'][0]['longest_stationary_ms'], 3000)

    def test_scattered_stationary_time_rejected_even_without_long_pause(self):
        x = 0
        frames = self.moving(0)
        for t in range(0, 10000, 50):
            frames.append(frame(1, t, x))
            # 75% stationary, with each stationary stretch under three seconds.
            if t % 2000 >= 1500:
                x += 1
        self.write(frames, [event('spawn', 0, 0), event('spawn', 1, 0)])
        report = imp.import_source(self.source, self.output)
        self.assertEqual(report['rejected'], {'too_much_stationary_time': 1})
        rejected = next(c for c in report['quality']['clips'] if not c['accepted'])
        self.assertLess(rejected['longest_stationary_ms'], 3000)
        self.assertGreater(rejected['stationary_fraction'], .60)

    def test_stationary_clip_has_all_failures_but_counts_only_once(self):
        frames = self.moving(0) + [frame(1, t, 0) for t in range(0, 10000, 50)]
        self.write(frames, [event('spawn', 0, 0), event('spawn', 1, 0)])
        report = imp.import_source(self.source, self.output)
        self.assertEqual(report['quality']['rejected'], 1)
        self.assertEqual(sum(report['rejected'].values()), 1)
        self.assertEqual(report['quality']['failure_counts'],
                         {'stationary_stretch_too_long': 1, 'too_much_stationary_time': 1})

    def test_accepted_binary_samples_actions_and_ids_are_unchanged(self):
        self.write(self.moving(), [event('spawn', 0, 0), event('reload', 0, 3500), event('shot', 0, 6500)])
        report = imp.import_source(self.source, self.output)
        path = self.output / report['libraries'][0]['file']
        expected = path.read_bytes()
        relaxed = self.root / 'relaxed'
        report2 = imp.import_source(self.source, relaxed, min_duration=1,
                                    max_stationary_ms=0, max_stationary_fraction=1)
        self.assertEqual((relaxed / report2['libraries'][0]['file']).read_bytes(), expected)
        clip = decode(path)['clips'][0]
        self.assertEqual(clip['frames'][0][0], 0)
        self.assertEqual(clip['duration'], 10000)
        self.assertEqual(clip['actions'], [(3500, 1), (6500, 2)])

    def test_empty_spawn_pool_is_reported_not_replaced_with_other_spawn(self):
        self.write(self.moving() + self.moving(1, 500, 100),
                   [event('spawn', 0, 0), event('spawn', 1, 0, x=100)])
        report = imp.import_source(self.source, self.output)
        empty = next(p for p in report['quality']['pools'] if p['accepted'] == 0)
        self.assertEqual((empty['before'], empty['rejected'], empty['spawn']), (1, 1, (100, 0, 1)))
        self.assertEqual(len(report['libraries'][0]['pools']), 1)
        saved = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(saved['quality']['rejected'], 1)

    def test_all_rejected_explains_reason_and_leaves_output_unchanged(self):
        self.write()
        self.output.mkdir()
        old = self.output / 'manifest.json'
        old.write_text('existing manifest')
        with self.assertRaisesRegex(ValueError, 'life_too_short.*existing output unchanged'):
            imp.import_source(self.source, self.output)
        self.assertEqual(old.read_text(), 'existing manifest')
        self.assertEqual(list(self.output.iterdir()), [old])

    def test_legacy_thresholds_can_be_requested_explicitly(self):
        self.write([frame(0, t, 0) for t in range(0, 500, 50)], [event('spawn', 0, 0)])
        report = imp.import_source(self.source, self.output, min_duration=500,
                                   max_stationary_ms=0, max_stationary_fraction=1)
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['quality']['clips'][0]['stationary_fraction'], 1)

    def test_duplicate_rejected_clip_is_not_counted_twice_in_quality_report(self):
        frames = self.moving() + self.moving(1, 500)
        events = [event('spawn', 0, 0), event('spawn', 1, 0)]
        self.write(frames, events)
        self.write(frames, events, folder='copy')
        report = imp.import_source(self.source, self.output)
        self.assertEqual(report['quality']['candidates'], 2)
        self.assertEqual(report['quality']['rejected'], 1)
        self.assertEqual(report['rejected']['duplicate_clip'], 2)

    def test_cli_defaults_and_empty_pool_warning(self):
        self.write(self.moving() + self.moving(1, 500, 100),
                   [event('spawn', 0, 0), event('spawn', 1, 0, x=100)])
        result = subprocess.run([sys.executable, imp.__file__, str(self.source), str(self.output)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('retained 1/2', result.stdout)
        self.assertIn('WARNING: no eligible lives left', result.stderr)
        self.assertIn('100.0', result.stderr)

    def test_cli_rejects_nan_settings(self):
        result = subprocess.run([sys.executable, imp.__file__, str(self.source), str(self.output),
                                 '--stationary-speed', 'nan'], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('stationary_speed must be finite', result.stderr)

    def test_stale_library_for_fully_filtered_map_is_not_silently_reused(self):
        self.write(self.moving(), [event('spawn', 0, 0)])
        self.write([frame(0, 0, 0, map='dm/short'), frame(0, 50, 1, map='dm/short')],
                   [event('spawn', 0, 0)], folder='other', map_name='dm/short')
        stale = self.output / 'dm/short.4294967295.1.rpl'
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b'existing library')
        with self.assertRaisesRegex(ValueError, 'old library exists'):
            imp.import_source(self.source, self.output)
        self.assertEqual(stale.read_bytes(), b'existing library')
        self.assertFalse((self.output / 'dm/test.4294967295.1.rpl').exists())


if __name__=='__main__':
    unittest.main()
