"""Historical test entry, now verifying the current anchored-goal binary contract."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy_entry import validate_metadata
from plus_protocol import VERSION,REPRESENTATION
from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface as adapter
from examples.simBenchmarks.LIBERO.eval_files.eval_libero import _resolve_gripper_command
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import inference_seed

class Peer:
    def __init__(self,*a,**kw):self.calls=[]
    def get_server_metadata(self):return dict(action_representation=REPRESENTATION,
        action_contract={'version':VERSION},action_chunk_size=16,available_unnorm_keys=['franka'])
    def predict_action(self,request):
        self.calls.append(request)
        actions=np.zeros((1,16,16),dtype=np.float32);actions[:,:,0]=.01;actions[:,:,6]=1
        actions[:,:,7]=np.arange(16)%2
        return {'data':{'actions':actions}}

class GoalTests(unittest.TestCase):
    def test_latest_metadata_passes_old_width_metadata_rejected(self):
        validate_metadata(Peer().get_server_metadata())
        with self.assertRaises(ValueError):validate_metadata(dict(action_representation='libero_umi16',action_chunk_size=16))
    def test_fixed_chunk_anchor_native_reference_binary_gripper_and_isolation(self):
        common=dict(eef_quat=[0,0,0,1],controller_eef_quat=[0,0,0,1],controller_goal_quat=[0,0,0,1])
        example={'lang':'goal','image':[np.zeros((256,256,3),dtype=np.uint8)]*2}
        with patch.object(adapter,'WebsocketClientPolicy',Peer):
            first=adapter.ModelClient(action_interface='anchored_goal_binary')
            second=adapter.ModelClient(action_interface='anchored_goal_binary')
            a=first.step(example,step=0,eef_pos=[0,0,0],controller_eef_pos=[0,0,0],diagnostic_seed=77,**common)['raw_action']
            b=first.step(example,step=1,eef_pos=[.005,0,0],controller_eef_pos=[.005,0,0],**common)['raw_action']
            second.step(example,step=0,eef_pos=[1,0,0],controller_eef_pos=[1,0,0],diagnostic_seed=88,**common)
            np.testing.assert_allclose(a['world_vector'],[.2,0,0],atol=1e-6)
            np.testing.assert_allclose(b['world_vector'],[.1,0,0],atol=1e-6)
            np.testing.assert_array_equal(_resolve_gripper_command(a),[1])
            np.testing.assert_array_equal(_resolve_gripper_command(b),[-1])
            self.assertEqual(len(first.client.calls),1)
            self.assertEqual(first.client.calls[0]['diagnostic_seed'],77)
            self.assertNotIn('state',first.client.calls[0]['examples'][0])
            np.testing.assert_array_equal(first._anchor_pos,[0,0,0])
            np.testing.assert_array_equal(second._anchor_pos,[1,0,0])
            first.reset('goal');self.assertIsNone(first.raw_actions);self.assertIsNone(first._anchor_pos)
    def test_seed_depends_on_variant_and_chunk_not_worker(self):
        self.assertEqual(inference_seed(7,'libero_goal',34,0,3),inference_seed(7,'libero_goal',34,0,3))
        self.assertNotEqual(inference_seed(7,'libero_goal',34,0,3),inference_seed(7,'libero_goal',35,0,3))
        self.assertNotEqual(inference_seed(7,'libero_goal',34,0,3),inference_seed(7,'libero_goal',34,0,4))
if __name__=='__main__':unittest.main()
