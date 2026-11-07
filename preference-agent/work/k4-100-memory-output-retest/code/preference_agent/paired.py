"""Same-controller history ablation with independent checkpointed arm results."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from .core import FeedbackMiner, Pipeline
from .react_critic import ForecastValidationError

class PairedExecutionError(RuntimeError):
    """Fatal run error; safe partial results survive without exposing SDK exceptions."""
    def __init__(self, stage, error, partial_result):
        super().__init__(f'Paired run stopped at {stage}: {type(error).__name__}')
        self.stage = stage
        self.cause_type = type(error).__name__
        self.partial_result = deepcopy(partial_result)

class PairedEvaluator:
    def __init__(self, model):
        if getattr(model, 'critic_mode', None) != 'react':
            raise ValueError('Paired evaluator requires the same ReAct model for both arms')
        self.pipeline = Pipeline(model)
        self.memory_cache = {}

    def run(self, history, repo, target_order, context, response, *, on_update=None):
        past = FeedbackMiner().run([e for e in history if e.repo == repo and e.order < target_order])
        snapshot = json.dumps([asdict(e) for e in past], sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256((repo + '\n' + snapshot).encode()).hexdigest()
        reused = key in self.memory_cache
        result = {'no_memory': None, 'agent': None, 'status': 'running',
                  'arm_status': {'no_memory': 'pending', 'agent': 'pending'}, 'errors': {},
                  'protocol': {'controller': 'react', 'memory_key': key, 'memory_reused': reused,
                               'history_events': len(past), 'revision_enabled': False}}
        def checkpoint():
            if on_update is not None: on_update(deepcopy(result))
        def fatal(stage, error):
            result['status'] = 'stopped'
            result['errors'][stage] = {'error_type': type(error).__name__, 'fatal': True}
            if getattr(error, 'forecast_diagnostics', None) is not None:
                result['errors'][stage]['diagnostics'] = deepcopy(error.forecast_diagnostics)
            checkpoint()
            raise PairedExecutionError(stage, error, result) from None
        def arm(name, events, rules):
            try:
                result[name] = self.pipeline.predict_with_memory(events, rules, repo, target_order,
                                                                  context, response, revise=False)
                result['arm_status'][name] = 'ok'
            except ForecastValidationError as error:
                result['arm_status'][name] = 'failed'
                result['errors'][name] = {'error_type': type(error).__name__, 'message': str(error),
                                         'diagnostics': deepcopy(error.diagnostics), 'fatal': False}
            except Exception as error:
                result['arm_status'][name] = 'failed'
                fatal(name, error)
            checkpoint()
        # Save the independent baseline before attempting any memory work.
        arm('no_memory', [], [])
        if not reused:
            try:
                self.memory_cache[key] = self.pipeline.build_memory(past, repo, target_order)
            except Exception as error:
                fatal('memory', error)
        arm('agent', past, self.memory_cache[key])
        succeeded = sum(status == 'ok' for status in result['arm_status'].values())
        result['status'] = 'ok' if succeeded == 2 else 'partial' if succeeded else 'failed'
        checkpoint()
        return result
