import unittest
from preference_agent.core import Event,Rule,Pipeline
from preference_agent.demo import DemoModel

class MemoryReuseTests(unittest.TestCase):
    def test_cached_memory_matches_regular_pipeline_and_is_not_mutated(self):
        model=DemoModel();pipeline=Pipeline(model)
        history=[Event(str(i),'r',i,'','change','run tests','correction') for i in (1,2)]
        memory=pipeline.build_memory(history,'r',3)
        cached=pipeline.predict_with_memory(history,memory,'r',3,'tests','untested')
        self.assertEqual(cached,pipeline.run(history,'r',3,'tests','untested'))
        cached['memory'][0]['evidence'].append('mutated')
        self.assertNotIn('mutated',memory[0].evidence)

    def test_cached_future_and_foreign_memory_rejected(self):
        pipeline=Pipeline(DemoModel())
        for rule in [Rule('x','foreign'),Rule('x','r',['future'])]:
            with self.assertRaisesRegex(ValueError,'out-of-scope'):
                pipeline.predict_with_memory([], [rule], 'r',1,'','')
