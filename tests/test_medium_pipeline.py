import hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/train_medium_pipeline.py'

class MediumPipelineTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('medium_pipeline_test',SCRIPT)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        m=self.module;m.ROOT=Path(self.tmp.name);(m.ROOT/'reports').mkdir();m.STATE=m.ROOT/'reports/job.json'
        m.STOP=m.ROOT/'reports/stop';m.BASE=m.ROOT/'base';m.SFT_ROOT=m.ROOT/'sft';m.TARGET=m.ROOT/'candidate';m.PLANNED_STEPS=500
        self.steps=0;self.stop_after_first=False;self.pretrain_calls=0;self.train_paths=[]

    def fake_pretrain(self,*args,**kwargs):
        m=self.module;self.pretrain_calls+=1;self.steps+=250;m.BASE.mkdir(exist_ok=True)
        (m.BASE/'latest.json').write_text('{}')
        (m.BASE/'training.json').write_text(json.dumps({'completed_steps':self.steps,'tokens_seen':self.steps*2048,'status':'complete' if self.steps==500 else 'checkpointed'}))
        if self.stop_after_first:m.STOP.touch()

    def fake_train(self,data,output,**kwargs):
        p=Path(output);p.mkdir(parents=True);self.train_paths.append(p)
        (p/'training.json').write_text('{"status":"complete"}');(p/'model.safetensors').write_bytes(b'test-only-weights')

    def fake_subprocess(self,argv,**kwargs):
        m=self.module;source=Path(argv[argv.index('--run')+1]);digest=hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest()
        target=m.ROOT/argv[argv.index('--output')+1]
        if 'export_dialogue.py' in argv[1]:
            target.mkdir();(target/'training.json').write_text(json.dumps({'source_weights_sha256':digest}))
        else:target.write_text(json.dumps({'model_sha256':digest,'total_pass':0,'total':95}))
        return MagicMock(returncode=0)

    def run_pipeline(self,resume=False):
        m=self.module
        with patch.object(sys,'argv',[str(SCRIPT)]+(['--resume'] if resume else [])),patch('os.chdir'),patch('os.umask'),patch.object(m,'active_pid',return_value=False),patch('torch.backends.mps.is_available',return_value=True),patch('torch.mps.empty_cache'),patch('trid.pretraining.pretrain',side_effect=self.fake_pretrain),patch('trid.training.train',side_effect=self.fake_train),patch.object(m.subprocess,'Popen') as popen,patch.object(m.subprocess,'run',side_effect=self.fake_subprocess):
            popen.return_value.poll.return_value=None;m.main();popen.return_value.terminate.assert_called_once()
        return json.loads(m.STATE.read_text())

    def test_bounded_pipeline_evaluates_without_activating(self):
        state=self.run_pipeline()
        self.assertEqual(self.pretrain_calls,2);self.assertEqual(state['status'],'complete')
        self.assertEqual(state['stage'],'review_required');self.assertFalse(state['auto_activate'])
        self.assertEqual(state['pretrain_tokens_seen'],500*2048)
        self.assertFalse((self.module.ROOT/'active-model.json').exists())

    def test_stop_after_checkpoint_does_not_start_sft(self):
        self.stop_after_first=True;state=self.run_pipeline()
        self.assertEqual(self.pretrain_calls,1);self.assertEqual(state['status'],'stopped');self.assertEqual(self.train_paths,[])

    def test_resume_preserves_interrupted_sft_attempt(self):
        m=self.module;m.BASE.mkdir();(m.BASE/'training.json').write_text('{"completed_steps":500,"status":"complete"}')
        old=m.SFT_ROOT/'attempt-001';old.mkdir(parents=True);(old/'training.json').write_text('{"status":"training"}')
        m.STATE.write_text('{"pid":-1,"status":"interrupted"}')
        state=self.run_pipeline(resume=True)
        self.assertEqual(self.pretrain_calls,0);self.assertEqual(self.train_paths,[m.SFT_ROOT/'attempt-002'])
        self.assertEqual(json.loads((old/'training.json').read_text())['status'],'training');self.assertEqual(state['status'],'complete')

if __name__=='__main__':unittest.main()
