"""Deterministic plumbing demo. This backend does not learn preferences."""
import json
from .core import Event, Pipeline

class DemoModel:
    def explain(self, event): return 'Run tests before claiming a code change is verified.'
    def verify(self, rule, event):
        verdict = 'support' if event.label == 'pushback' and 'test' in event.follow_up.lower() else 'irrelevant'
        return {'verdict': verdict, 'reason': 'Synthetic feedback check',
                'feedback_excerpt': event.follow_up if verdict == 'support' else ''}
    def predict(self, context, response, rules):
        risk = bool(rules) and 'untested' in response.lower()
        return {'pushback_probability': .9 if risk else .3,
                'rationale': 'Synthetic demonstration values; not a trained forecast.'}
    def revise(self, context, response, prediction):
        return response + '\nI will run the relevant tests before claiming verification.'

def main():
    history = [Event(str(i), 'example', i, 'Fix code', 'Done',
                     'Please run tests first', 'pushback') for i in (1, 2)]
    result = Pipeline(DemoModel()).run(history, 'example', 3,
                                     'Fix code and run tests', 'The change is untested.')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
