"""Synthetic contract and causal-alignment regression tests. GPL-2.0-or-later."""
import csv,io,json,math,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import features as f
import prepare


def row(i=0,**kwargs):
    r=dict(schema=1,session_id='test',client_msec=i*20,server_msec=1000+i*20,cmd_number=i+1,cmd_server_msec=1000+i*20,
           forwardmove=127,rightmove=0,upmove=0,buttons=4,pm_type=0,pm_flags=0,health=100,team=4,walking=1,
           origin_x=i,origin_y=0,origin_z=0,velocity_x=50,velocity_y=0,velocity_z=0,view_pitch=0,view_yaw=i*2,view_roll=0,
           lean_angle=0,viewheight=82,clip_ammo=32,ammo=100,weapon_name='MP40',
           target_visible=0,target_confidence=0,target_angular_error=-1,target_distance=-1,
           target_relative_forward=0,target_relative_right=0,target_relative_up=0,target_velocity_x=0,target_velocity_y=0,target_velocity_z=0)
    r.update(kwargs);return r

def csv_text(rows):
    o=io.StringIO();w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return o.getvalue()

class ContractTests(unittest.TestCase):
    def test_dimensions(self):self.assertEqual(f.features(row()).shape,(54,))
    def test_angle_wrap(self):self.assertEqual(f.angle_delta(179,-179),2)
    def test_joint_xy(self):self.assertEqual(f.action_classes(127,-127,0,4|32|1),(6,1,2,3))
    def test_conflicting_lean(self):self.assertEqual(f.action_classes(0,0,0,16|32)[2],1)
    def test_team_neutral(self):np.testing.assert_array_equal(f.features(row(team=3)),f.features(row(team=4)))
    def test_clock_neutral(self):np.testing.assert_array_equal(f.features(row(server_msec=900000)),f.features(row()))
    def test_weapon_conditioning(self):self.assertFalse(np.array_equal(f.features(row(weapon_name='MP40')),f.features(row(weapon_name='KAR98 - Sniper'))))
    def test_hidden_targets_zero(self):self.assertTrue(np.all(f.features(row(target_relative_right=100,target_distance=10))[46:]==0))
    def test_target_admission(self):
        a=f.features(row(target_visible=1,target_confidence=1,target_angular_error=12,target_distance=100,target_relative_forward=99))
        self.assertEqual(a[46],1)
        a=f.features(row(target_visible=1,target_confidence=1,target_angular_error=12.01,target_distance=100))
        self.assertEqual(a[46],0)
    def test_nonfinite(self):
        with self.assertRaises(ValueError):f.features(row(origin_x=float('nan')))
    def test_spectator_exclusion(self):self.assertFalse(prepare.eligible(row(pm_flags=4)))
    def test_dead_exclusion(self):self.assertFalse(prepare.eligible(row(health=0)))
    def test_short_pause_retained(self):self.assertTrue(prepare.eligible(row(forwardmove=0,rightmove=0,buttons=4)))
    def test_causal_view_labels_and_group_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'source.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for group in range(6):
                    sid=f'G{group}_maps_dm_mohdm6_bsp_1';base=f't/{sid}'
                    z.writestr(base+'_meta.txt',f'schema=1\nsession_id={sid}\nsource=client_predicted\nmap={f.MAP}\nmap_checksum={f.CHECKSUM}\ntarget_game=0\n')
                    rows=[row(i,session_id=sid) for i in range(8)]
                    z.writestr(base+'_frames.csv',csv_text(rows))
                    # Mouse-only changes are absent from input log, as in the uploaded files.
                    inp=dict(cmd_number=1,client_msec=0,cmd_server_msec=1000,cmd_pitch=0,cmd_yaw=0,
                             forwardmove=127,rightmove=0,upmove=0,buttons=4)
                    z.writestr(base+'_inputs.csv',csv_text([inp]))
            prepare.prepare(archive,root/'data')
            report=json.loads((root/'data/dataset-report.json').read_text())
            groups=[]
            for split in ('train','validation','test'):
                data=np.load(root/'data'/f'{split}.npz');self.assertTrue(np.all(data['turn'][:,1]==2))
                self.assertTrue(np.all(data['categories'][:,0]==7))
                self.assertEqual(len(data['x']),14)
                groups.append(set(report['splits'][split]['groups']))
            self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    def test_no_runtime_position_writes(self):
        source=Path(__file__).resolve().parents[2]/'code/fgame/g_imitation.cpp'
        text=source.read_text()
        for forbidden in ('setOrigin(', 'setContents(', 'MOVETYPE_NONE', 'G_Replay', 'Respawn('):
            self.assertNotIn(forbidden,text)

if __name__=='__main__':unittest.main()
