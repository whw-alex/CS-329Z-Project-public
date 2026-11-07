import os
import unittest
from unittest.mock import patch
from preference_agent.dspy_backend import configure_nvidia, configure_openrouter

class OpenRouterConfigTests(unittest.TestCase):
    def test_provider_route_and_no_configuration_request(self):
        for model in ['google/example-model', 'openrouter/google/example-model']:
            with self.subTest(model=model), patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-only', 'OPENROUTER_MODEL': model}, clear=True), patch('preference_agent.dspy_backend.dspy.LM') as lm, patch('preference_agent.dspy_backend.dspy.configure') as configure:
                self.assertEqual(configure_openrouter(), 'openrouter/google/example-model')
                lm.assert_called_once_with('openrouter/google/example-model', api_key='test-only', api_base='https://openrouter.ai/api/v1', cache=False, num_retries=0, timeout=60, max_tokens=2048)
                configure.assert_called_once()

    def test_explicit_model_overrides_environment(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-only', 'OPENROUTER_MODEL': 'google/old'}, clear=True), patch('preference_agent.dspy_backend.dspy.LM'), patch('preference_agent.dspy_backend.dspy.configure'):
            self.assertEqual(configure_openrouter('google/new'), 'openrouter/google/new')

    def test_missing_configuration_fails_before_lm(self):
        for values in [{}, {'OPENROUTER_API_KEY': 'test-only'}, {'OPENROUTER_MODEL': 'google/example'}]:
            with self.subTest(values=list(values)), patch.dict(os.environ, values, clear=True), patch('preference_agent.dspy_backend.dspy.LM') as lm:
                with self.assertRaises(ValueError): configure_openrouter()
                lm.assert_not_called()

    def test_invalid_model_is_rejected(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-only'}, clear=True), patch('preference_agent.dspy_backend.dspy.LM') as lm:
            with self.assertRaises(ValueError): configure_openrouter('model-without-provider')
            lm.assert_not_called()

class NvidiaConfigTests(unittest.TestCase):
    def test_provider_route_default_base_and_no_configuration_request(self):
        for model in ['meta/example-model', 'nvidia_nim/meta/example-model']:
            with self.subTest(model=model), patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-only', 'NVIDIA_MODEL': model}, clear=True), patch('preference_agent.dspy_backend.dspy.LM') as lm, patch('preference_agent.dspy_backend.dspy.configure') as configure:
                self.assertEqual(configure_nvidia(), 'nvidia_nim/meta/example-model')
                lm.assert_called_once_with('nvidia_nim/meta/example-model', api_key='test-only', api_base='https://integrate.api.nvidia.com/v1', cache=False, num_retries=0, timeout=60, max_tokens=2048)
                configure.assert_called_once()

    def test_self_hosted_base_and_explicit_model(self):
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-only', 'NVIDIA_MODEL': 'meta/old', 'NVIDIA_API_BASE': 'http://localhost:8000/v1'}, clear=True), patch('preference_agent.dspy_backend.dspy.LM') as lm, patch('preference_agent.dspy_backend.dspy.configure'):
            self.assertEqual(configure_nvidia('meta/new'), 'nvidia_nim/meta/new')
            self.assertEqual(lm.call_args.kwargs['api_base'], 'http://localhost:8000/v1')

    def test_missing_or_invalid_configuration_fails_before_lm(self):
        for values, model in [({}, None), ({'NVIDIA_API_KEY': 'test-only'}, None), ({'NVIDIA_MODEL': 'meta/x'}, None), ({'NVIDIA_API_KEY': 'test-only'}, 'no-publisher')]:
            with self.subTest(values=list(values), model=model), patch.dict(os.environ, values, clear=True), patch('preference_agent.dspy_backend.dspy.LM') as lm:
                with self.assertRaises(ValueError): configure_nvidia(model)
                lm.assert_not_called()

class EnvironmentFileTests(unittest.TestCase):
    def test_cli_loads_explicit_file_without_overriding_or_interpolation(self):
        import json
        import tempfile
        from pathlib import Path
        from preference_agent.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            env_file = directory / '.env'
            env_file.write_text('OPENROUTER_API_KEY=file-value\nOPENROUTER_MODEL=google/example\nCONFIG_LITERAL=${HOME}\n')
            case = directory / 'case.json'
            case.write_text(json.dumps({'history': [], 'target': {'repo': 'repo', 'target_order': 1, 'context': 'task', 'response': 'response'}}))
            output = directory / 'result.json'
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'shell-value'}, clear=True), patch('sys.argv', ['preference_agent', str(case), '--env-file', str(env_file), '--output', str(output)]):
                main()
                self.assertEqual(os.environ['OPENROUTER_API_KEY'], 'shell-value')
                self.assertEqual(os.environ['OPENROUTER_MODEL'], 'google/example')
                self.assertEqual(os.environ['CONFIG_LITERAL'], '${HOME}')
                self.assertEqual(json.loads(output.read_text())['backend'], 'demo')
