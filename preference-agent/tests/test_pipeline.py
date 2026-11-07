import unittest
from preference_agent.core import Event, Pipeline
from preference_agent.demo import DemoModel

class PipelineTests(unittest.TestCase):
    def test_cutoff_and_repo_isolation(self):
        events = [Event(str(i), repo, order, '', '', 'run tests', 'correction')
                  for i, repo, order in [(1, 'a', 1), (2, 'a', 2), (3, 'a', 3), (4, 'b', 1)]]
        result = Pipeline(DemoModel()).run(events, 'a', 3, 'run tests', 'untested', revise=True)
        self.assertEqual(result['history_ids'], ['1', '2'])
        self.assertEqual(result['memory'][0]['evidence'], ['1', '2'])
        self.assertEqual(result['memory'][0]['status'], 'active')
        self.assertIn('run the relevant tests', result['revised_response'])

    def test_single_event_not_verified_by_itself(self):
        event = Event('1', 'a', 1, '', '', 'run tests', 'correction')
        result = Pipeline(DemoModel()).run([event], 'a', 2, 'tests', 'untested')
        self.assertEqual(result['memory'][0]['status'], 'candidate')
        self.assertEqual(result['retrieved_rules'], [])

if __name__ == '__main__': unittest.main()
