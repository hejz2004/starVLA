"""Real subprocess checks for current checkpoint admission and safe resource reuse."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import yaml
from plus_protocol import ROOT,VERSION
BUNDLE=Path(__file__).resolve().parents[1]

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.run=self.root/'run';(self.run/'preflight').mkdir(parents=True)
        source=self.root/'source';bench=source/'libero/libero'
        for p in [bench/'benchmark',bench/'bddl_files',bench/'init_files']:p.mkdir(parents=True,exist_ok=True)
        (bench/'benchmark/task_classification.json').write_bytes((BUNDLE/'launchers/task_classification.json').read_bytes())
        assets=self.root/'assets';(assets/'scenes/tabletop250').mkdir(parents=True)
        (assets/'scenes/tabletop250/tabletop_table_FabricCarpet001_NRM_3K.xml').write_text('fixture')
        base=self.root/'base';(base/'articulated_objects').mkdir(parents=True)
        (base/'articulated_objects/wooden_cabinet.xml').write_text('fixture')
        model=self.root/'model';(model/'checkpoints').mkdir(parents=True)
        ckpt=model/'checkpoints/steps_1.pt';ckpt.write_bytes(b'checkpoint-fixture')
        self.cfg=dict(framework=dict(name='DualStreamQwenPI',action_model=dict(action_dim=16,action_horizon=16,num_inference_timesteps=4,use_state=False)),datasets=dict(vla_data=dict(action_type=VERSION)))
        self.config=model/'config.yaml';self.config.write_text(yaml.safe_dump(self.cfg))
        (model/'dataset_statistics.json').write_text('{"franka":{}}')
        self.env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',RUN_ROOT=str(self.run),
            STARVLA_DIR=str(ROOT),CKPT_PATH=str(ckpt),LIBERO_HOME=str(source),LIBERO_ASSETS=str(assets),LIBERO_BASE_ASSETS=str(base))
    def tearDown(self):self.tmp.cleanup()
    def execute(self):return subprocess.run([sys.executable,str(BUNDLE/'preflight.py'),'inputs'],env=self.env,capture_output=True,text=True)
    def test_resource_links_are_idempotent_for_resume(self):
        first=self.execute();self.assertEqual(first.returncode,0,first.stderr)
        second=self.execute();self.assertEqual(second.returncode,0,second.stderr)
        self.assertTrue((self.run/'asset_links/articulated_objects/wooden_cabinet.xml').is_file())
        self.assertTrue((self.run/'libero_source/libero/libero/assets').is_symlink())
    def test_old_width_checkpoint_rejected_before_model_loading(self):
        self.cfg['datasets']['vla_data']['action_type']='libero_umi16'
        self.config.write_text(yaml.safe_dump(self.cfg))
        r=self.execute();self.assertNotEqual(r.returncode,0);self.assertIn('anchored-goal binary',r.stderr)
    def test_changed_resource_pointer_is_rejected_on_resume(self):
        self.assertEqual(self.execute().returncode,0)
        target=self.run/'asset_links/articulated_objects';target.unlink()
        other=self.root/'wrong';other.mkdir();target.symlink_to(other)
        r=self.execute();self.assertNotEqual(r.returncode,0);self.assertIn('resource link changed',r.stderr)
if __name__=='__main__':unittest.main()
