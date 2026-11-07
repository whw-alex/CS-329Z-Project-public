import unittest
try:
    import numpy as np
    import pandas as pd
    from preference_agent.evaluate import metrics,prepare,paired_bootstrap
except ModuleNotFoundError as error:
    if error.name not in {'numpy','pandas','pyarrow','sklearn'}:raise
    pd=None

@unittest.skipIf(pd is None,'Install evaluation/data dependencies')
class EvalTests(unittest.TestCase):
    def fixture(self):
        t=pd.DataFrame([
            dict(session_id='a0',turn_number=2,repo_id='a',rank=0,n_sess=2,split='train',prompt_pushback='rejection',t1=1),
            dict(session_id='a1',turn_number=2,repo_id='a',rank=1,n_sess=2,split='train',prompt_pushback='non_pushback',t1=0),
            dict(session_id='b0',turn_number=2,repo_id='b',rank=0,n_sess=2,split='holdout',prompt_pushback='correction',t1=1),
            dict(session_id='b1',turn_number=2,repo_id='b',rank=1,n_sess=2,split='holdout',prompt_pushback='failure_report',t1=1),
        ])
        return t,pd.DataFrame([dict(repo_id='a',split='train'),dict(repo_id='b',split='holdout')])

    def test_only_train_targets_and_own_history_define_probabilities(self):
        t,s=self.fixture();frame,train,b0,b1,prior=prepare(t,s,1,'holdout')
        self.assertEqual(prior,0.0)  # the only eligible train target is non_pushback
        np.testing.assert_array_equal(b1,[1.0])  # repo b's first session is pushback
        self.assertEqual(len(train),1)
        self.assertEqual(frame.session_id.tolist(),['b1'])

    def test_eval_label_mutation_cannot_change_predictions(self):
        t,s=self.fixture();a=prepare(t,s,1,'holdout')
        t.loc[t.session_id.eq('b1'),['prompt_pushback','t1']]=['non_pushback',0]
        b=prepare(t,s,1,'holdout')
        np.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(a[3],b[3])

    def test_empty_history_falls_back_on_same_rows(self):
        t,s=self.fixture();t=t[t.session_id!='b0']
        frame,_,b0,b1,_=prepare(t,s,1,'holdout')
        self.assertFalse(frame.history_available.any())
        np.testing.assert_array_equal(b0,b1)

    def test_binary_metrics(self):
        r=metrics([1,0,1,0],[.9,.1,.8,.2])
        self.assertEqual(r['auroc'],1);self.assertEqual(r['f1_at_0_5'],1);self.assertLess(r['brier'],.05)
        with self.assertRaises(ValueError): metrics([1],[1.5])

    def test_single_class_auroc_is_undefined(self):
        r=metrics([1],[.9])
        self.assertIsNone(r['auroc'])

    def test_bootstrap_is_reproducible_and_paired(self):
        f=pd.DataFrame({'repo_id':['a','a','b','b'],'t1':[0,1,0,1]})
        p=np.full(4,.5)
        a=paired_bootstrap(f,p,p,10,0);b=paired_bootstrap(f,p,p,10,0)
        self.assertEqual(a,b)
        self.assertEqual(a['brier_B0_minus_B1']['ci95'],[0,0])
