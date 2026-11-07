import unittest
from preference_agent.sessions import Session, select_history
from preference_agent.core import Event

class SessionTests(unittest.TestCase):
    def sessions(self):
        return [Session(str(i), 'r', f'2026-01-0{i}T00:00:00Z', f'2026-01-0{i}T01:00:00Z',
                        [Event(str(i), 'r', i, '', '', '', 'no_pushback')]) for i in (1, 2, 3, 4)]

    def test_history_stays_fixed_for_later_targets(self):
        sessions = self.sessions()
        self.assertEqual([e.id for e in select_history(sessions, 'r', '3', 1)], ['1'])
        self.assertEqual([e.id for e in select_history(sessions, 'r', '4', 1)], ['1'])

    def test_target_cannot_be_in_history(self):
        with self.assertRaises(ValueError): select_history(self.sessions(), 'r', '1', 1)

    def test_overlap_rejected(self):
        sessions = self.sessions()
        sessions[0].ended_at = '2026-01-04T01:00:00Z'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            select_history(sessions, 'r', '3', 1)
