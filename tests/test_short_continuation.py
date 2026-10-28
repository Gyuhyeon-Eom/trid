import contextlib,io,json,tempfile,unittest,random
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import torch
from safetensors.torch import save_file,load_file
from trid.model import Config
from trid.short_reader import SpanModel,PresenceModel
from trid.text import train_tokenizer
from trid.corpus import write_json,sha256
from scripts.train_short_reader import train_component,sample_balanced_groups


class ShortContinuationTests(unittest.TestCase):
    def test_mixed_group_sizes_keep_exact_balance(self):
        groups=[[{'kind':i%2,'group':n} for i in range(size)] for n,size in enumerate([2,8,2,8])]
        rng=random.Random(42);seen=set()
        for _ in range(50):
            rows=sample_balanced_groups(groups,rng,16)
            self.assertEqual(len(rows),16);self.assertEqual(sum(r['kind'] for r in rows),8)
            seen.update(r['group'] for r in rows)
        self.assertEqual(seen,{0,1,2,3})

    def test_continue_both_components_preserves_parent_and_heads(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);data=root/'data';data.mkdir()
            for split,doc in [('train','식비는 3만원이다.'),('valid','식비는 4만원이다.')]:
                rs=[{'document':doc,'question':'식비는?','answer':doc[4:7],'is_impossible':False,'group':split},
                    {'document':doc,'question':'기관은?','answer':None,'is_impossible':True,'group':split}]
                (data/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
            write_json(data/'provenance.json',{'test':True})
            parent=root/'parent';parent.mkdir();tok=train_tokenizer([{'prompt':'식비는 3만원이다. 기관은?', 'answer':'3만원'}],parent/'tok.json',300)
            cfg=Config(vocab_size=tok.get_vocab_size(),dim=32,layers=1,heads=4,hidden_dim=64,context_length=256)
            for mode,cls in [('span',SpanModel),('presence',PresenceModel)]:
                path=parent/mode;path.mkdir();(path/'tokenizer.json').write_bytes((parent/'tok.json').read_bytes())
                write_json(path/'config.json',asdict(cfg));model=cls(cfg)
                save_file(model.state_dict(),str(path/'model.safetensors'));before=sha256(path/'model.safetensors')
                write_json(path/'training.json',{'stage':'short_reader_component_v1','component':mode,'status':'complete'})
                args=SimpleNamespace(output=str(root/'output'),data=str(data),init_components=str(parent),init_run='unused',seed=42,max_input_tokens=192,batch_size=2,steps=1,lr=1e-12,selection='loss')
                train_component(args,mode)
                self.assertEqual(before,sha256(path/'model.safetensors'))
                result=load_file(str(root/'output'/mode/'model.safetensors'))
                for k,v in model.state_dict().items():torch.testing.assert_close(result[k],v,atol=1e-8,rtol=1e-6)
                metadata=json.loads((root/'output'/mode/'training.json').read_text())
                self.assertTrue(metadata['continued_component']);self.assertEqual(metadata['initial_weights_sha256'],before)
