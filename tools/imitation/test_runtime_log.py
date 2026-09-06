"""Tests of runtime diagnostics, not trained behavior. SPDX-License-Identifier: GPL-2.0-or-later."""
import unittest
from audit_runtime_log import audit

def line(t,request=5,sent=4,history=0,permit=0,sequence=1):
    obs=[0.]*54;obs[33]=history
    return (f'imitation_frame bot=2 time={t} sequence={sequence} model_id=123 dt=20 reset=0 pitch=0 yaw=0 '
            f'requested=127,0,0,{request} sent_buttons={sent} visible=0 aligned=0 permitted={permit} obs='+','.join(map(str,obs)))
class LogTests(unittest.TestCase):
    def test_denied_fire_keeps_history(self):
        report=audit([line(20),line(40,history=1)])['sequences'][0]
        self.assertEqual(report['guard_denied_fire_ms'],40)
        self.assertNotIn('primary_intent_history_mismatches',report)
        self.assertNotIn('sent_fire_ms',report)
    def test_bad_history_is_reported(self):
        report=audit([line(20),line(40,history=0)])['sequences'][0]
        self.assertEqual(report['primary_intent_history_mismatches'],1)
    def test_real_spawn_separates_sequence(self):
        self.assertEqual(len(audit([line(20),line(40,sequence=2)])['sequences']),2)
    def test_bad_line_does_not_crash(self):
        self.assertEqual(audit(['imitation_frame foo=1'])['malformed_lines'],1)
    def test_no_bullets_inferred(self):
        report=audit([line(20,sent=5,permit=1)])['sequences'][0]
        self.assertEqual(report['sent_fire_ms'],20)
        self.assertNotIn('shots',report)
if __name__=='__main__':unittest.main()
