import copy,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import torch
from safetensors.torch import save_file
from trid.character_features import character_id,inherited_hash_indices,HANGUL_VOCAB_SIZE,RANGES
from trid.short_reader import SpanModel,load_component
from trid.model import Config
from trid.korean_reader import encode,batch
from trid.text import train_tokenizer
from trid.corpus import write_json


class HangulFeatureTests(unittest.TestCase):
    def test_hangul_ascii_and_jamo_have_no_collisions(self):
        chars=[chr(code) for start,end in RANGES for code in range(start,end)]
        ids=[character_id(ch,'hangul_v1') for ch in chars]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(character_id('대'),character_id('팀'))
        self.assertNotEqual(character_id('대','hangul_v1'),character_id('팀','hangul_v1'))
        inherited=inherited_hash_indices();self.assertEqual(len(inherited),HANGUL_VOCAB_SIZE)
        for ch in chars+['🙂','漢']:
            self.assertEqual(inherited[character_id(ch,'hangul_v1')],ord(ch)%512)

    def test_migration_preserves_predictions_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tok=train_tokenizer([{'prompt':'담당 부서는 대외행정실이다. 다른 팀은 총무팀이다.','answer':'대외행정실'}],root/'tokenizer.json',300)
            cfg=Config(vocab_size=tok.get_vocab_size(),dim=32,heads=4,layers=1,hidden_dim=64,context_length=256)
            old=SpanModel(cfg).eval();new=copy.deepcopy(old);new.expand_hangul_characters()
            doc='담당 부서는 대외행정실이다. 다른 팀은 총무팀이다.';q='담당 부서는?'
            a=batch([encode(tok,doc,q,256)],tok.token_to_id('<|pad|>'),'cpu')
            b=batch([encode(tok,doc,q,256,character_encoding='hangul_v1')],tok.token_to_id('<|pad|>'),'cpu')
            torch.testing.assert_close(old(*a)[1],new(*b)[1],rtol=0,atol=0)
            write_json(root/'config.json',asdict(cfg));write_json(root/'training.json',{'stage':'short_reader_component_v1','component':'span','character_encoding':'hangul_v1'})
            save_file(new.state_dict(),str(root/'model.safetensors'))
            loaded=load_component(root,'cpu');self.assertEqual(loaded.character_encoding,'hangul_v1')
            torch.testing.assert_close(loaded(*b)[1],new(*b)[1],rtol=0,atol=0)
            # Updating the new '대' feature no longer changes '팀'.
            dae=character_id('대','hangul_v1');team=character_id('팀','hangul_v1')
            before=new.character.weight[team].detach().clone()
            new.character(torch.tensor([dae])).sum().backward()
            with torch.no_grad():new.character.weight.add_(new.character.weight.grad,alpha=-.1)
            torch.testing.assert_close(before,new.character.weight[team],rtol=0,atol=0)
