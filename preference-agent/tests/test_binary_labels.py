import json
import unittest

from preference_agent.core import Event, binary_label, model_event
from preference_agent.dspy_backend import Forecast, compact_turn
from preference_agent.history_tools import HistoryTools

FOUR_CLASS = {'correction', 'rejection', 'failure_report'}

def events():
    return [Event(f's{i}:1', 'r', i, 'fix tests', 'done', 'run tests', label)
            for i, label in enumerate(['correction', 'rejection', 'failure_report', 'no_pushback'], 1)]

class BinaryLabelTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual({binary_label(e.label) for e in events()}, {'pushback', 'no_pushback'})
        self.assertEqual(model_event(events()[0])['label'], 'pushback')
        self.assertEqual({e.label for e in events()}, {'pushback', 'no_pushback'})  # legacy labels load as binary
        self.assertEqual(binary_label('non_pushback'), 'no_pushback')

    def test_model_facing_views_are_binary(self):
        history = events()
        tools = HistoryTools(history, [], 'r', 10, 'fix tests', 'done', max_calls=10)
        seen = [json.loads(tools.search_history('tests')['content'])]
        seen += [json.loads(tools.get_episode(e.id)['content']) for e in history]
        seen += [compact_turn(e) for e in history]
        labels = {item['label'] for group in seen for item in (group if isinstance(group, list) else [group])}
        self.assertEqual(labels, {'pushback', 'no_pushback'})

    def test_forecast_prompts_use_the_relabeling_standard(self):
        from preference_agent.baselines import DirectForecast
        from preference_agent.pushback_definition import pushback_rules
        rules = pushback_rules()
        self.assertIn('would the developer have needed to send this', rules)
        self.assertNotIn('### Step', rules)  # labeler steps stay out
        for signature in (Forecast, DirectForecast):
            self.assertIn(rules, signature.instructions)
            self.assertIn('NEXT message AFTER', signature.instructions)

if __name__ == '__main__': unittest.main()
