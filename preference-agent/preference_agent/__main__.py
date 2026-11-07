import argparse
import json
from pathlib import Path
from .core import Event, Pipeline, save_result

def main():
    parser = argparse.ArgumentParser(description='Run preference forecasting on a JSON case')
    parser.add_argument('input', type=Path)
    parser.add_argument('--backend', choices=['demo', 'gemini', 'dspy', 'openrouter', 'nvidia'], default='demo')
    parser.add_argument('--model', help='Model ID; otherwise use GEMINI_MODEL, OPENROUTER_MODEL or NVIDIA_MODEL')
    parser.add_argument('--env-file', type=Path, help='Explicit local dotenv file; existing environment variables take precedence')
    parser.add_argument('--history-mode', choices=['provided', 'none'], default='provided', help='none removes history for a same-controller ablation')
    parser.add_argument('--max-calls', type=int, default=100)
    parser.add_argument('--critic-mode', choices=['fixed', 'react'], default='react')
    parser.add_argument('--memory-mode', choices=['event', 'session'], default='event', help='session builds memory with a few calls per history session')
    parser.add_argument('--retrieval-policy', choices=['autonomous', 'search_first'], default='autonomous', help='search_first spends one tool call on historical search before ReAct; no search with empty history')
    parser.add_argument('--max-citation-repairs', type=int, choices=[0, 1], default=1, help='At most one autonomous retry after unread citations; shares decision/tool budgets')
    parser.add_argument('--max-react-iters', type=int, default=8)
    parser.add_argument('--max-tool-calls', type=int, default=6)
    parser.add_argument('--tool-output-chars', type=int, default=12000)
    parser.add_argument('--paired', action='store_true', help='Run same-ReAct no-history and history arms; disables rewriting')
    parser.add_argument('--revise', action='store_true', help='Enable experimental rewriting')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.paired and (args.backend not in {'dspy', 'openrouter', 'nvidia'} or args.critic_mode != 'react' or args.revise or args.history_mode != 'provided'):
        parser.error('--paired requires DSPy/OpenRouter/NVIDIA ReAct with provided history and without --revise')
    if args.env_file is not None:
        if not args.env_file.is_file():
            parser.error('--env-file must point to an existing local file')
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False, interpolate=False)
    case = json.loads(args.input.read_text(encoding='utf-8'))
    target = dict(case['target'])
    if 'sessions' in case:
        from .sessions import Session, select_history
        sessions = [Session(**{**s, 'events': [Event(**{**e, 'label': 'no_pushback' if e['label'] == 'non_pushback' else e['label']}) for e in s['events']]}) for s in case['sessions']]
        history = select_history(sessions, target['repo'], target.pop('session_id'), case['k'])
    else:
        history = [Event(**{**e, 'label': 'no_pushback' if e['label'] == 'non_pushback' else e['label']}) for e in case['history']]
    if args.history_mode == 'none':
        history = []
    if args.backend in {'dspy', 'openrouter', 'nvidia'}:
        from .dspy_backend import DSPyAgent, configure_backend
        model_name = configure_backend(args.backend, args.model)
        agent = DSPyAgent(max_calls=args.max_calls, critic_mode=args.critic_mode,
                          max_react_iters=args.max_react_iters, max_tool_calls=args.max_tool_calls,
                          tool_output_chars=args.tool_output_chars, retrieval_policy=args.retrieval_policy, max_citation_repairs=args.max_citation_repairs,
                          memory_mode=args.memory_mode)
        model = agent.stages
    elif args.backend == 'gemini':
        from .gemini import GeminiModel
        model = GeminiModel(args.model, max_calls=args.max_calls)
    else:
        from .demo import DemoModel
        model = DemoModel()
    if args.paired:
        from .paired import PairedEvaluator
        result = PairedEvaluator(model).run(history=history, **target,
            on_update=lambda partial: save_result(partial, args.output))
    elif args.backend in {'dspy', 'openrouter', 'nvidia'}:
        result = agent(history=history, **target, revise=args.revise).toDict()
    else:
        result = Pipeline(model).run(history, **target, revise=args.revise)
    result['backend'] = args.backend
    result['history_mode'] = args.history_mode
    if args.backend == 'gemini':
        result['model'] = model.model
        result['api_calls'] = model.calls
    if args.backend in {'dspy', 'openrouter', 'nvidia'}:
        result['model'] = model_name
        result['stage_calls'] = model.calls
    save_result(result, args.output)
    if args.paired and result['status'] != 'ok':
        raise SystemExit(2)

if __name__ == '__main__': main()
