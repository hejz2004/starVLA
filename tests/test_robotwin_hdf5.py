"""Geometry, temporal alignment, native JPEG semantics, worker and statistics checks."""
import io
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from starVLA.dataloader.robotwin_hdf5 import (
    RobotwinHDF5Dataset, prepare_assets, poses_from_observation, decode_image,
    to_environment_actions, collate_fn, HORIZON,
)
from starVLA.dataloader.umi_pose import actions_to_current_frame


def source(tmp_path):
    directory = tmp_path/'dataset/task/aloha-agilex_clean_50'
    (directory/'data').mkdir(parents=True)
    (directory/'instructions').mkdir()
    n = 20
    t = np.arange(n)
    image = np.zeros((24,32,3),np.uint8)
    image[:] = [220,30,10]  # simulator RGB fed directly to legacy OpenCV encoder
    encoded = cv2.imencode('.jpg',image)[1].tobytes()
    with h5py.File(directory/'data/episode0.hdf5','w') as f:
        for h, sign in [('left',1),('right',-1)]:
            q = Rotation.from_euler('z', (sign*(np.pi/2+t*.01))[:,None]).as_quat()
            q[::3] *= -1  # exercise full-episode antipodal unwrapping
            pose = np.c_[t*.001,sign*t*.002,np.ones(n)*.2,q[:,[3,0,1,2]]]
            f[f'endpose/{h}_endpose'] = pose
            f[f'endpose/{h}_gripper'] = t/(n-1)
        for camera in ('head_camera','left_camera','right_camera'):
            f[f'observation/{camera}/rgb'] = np.array([encoded]*n,dtype=f'S{len(encoded)}')
    (directory/'instructions/episode0.json').write_text(json.dumps({'seen':['train instruction'],'unseen':['test instruction']}))
    assets = tmp_path/'assets.json'
    prepare_assets(tmp_path,assets,expected_tasks=1,expected_episodes=1)
    return assets


def test_geometry_and_t_plus_one(tmp_path):
    ds = RobotwinHDF5Dataset(source(tmp_path),return_metadata=True)
    assert len(ds)==4  # complete 17-row windows, not 5 including t
    sample = ds[0]
    assert sample['lang']=='train instruction'
    assert sample['robotwin_metadata']['target_indices']==list(range(1,17))
    state, future = ds.read_lowdim(0)
    actual = ds.normalizer.inverse(sample['action'])
    for start in (0,8):
        anchor = Rotation.from_quat(state[0,start+3:start+7])
        expected = anchor.inv().apply(future[:,start:start+3]-state[0,start:start+3])
        np.testing.assert_allclose(actual[:,start:start+3],expected,atol=1e-6)
        expected_q = (anchor.inv()*Rotation.from_quat(future[:,start+3:start+7])).as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(actual[:,start+3:start+7]).as_matrix(),expected_q,atol=1e-6)
        np.testing.assert_allclose(actual[:,start+7],future[:,start+7],atol=1e-6)
    same = actions_to_current_frame(state,state)
    np.testing.assert_allclose(same[0,:3],0,atol=1e-7)
    np.testing.assert_allclose(same[0,3:7],[0,0,0,1],atol=1e-7)
    assert np.linalg.norm(actual[0,:3])>1e-4
    env = to_environment_actions(state,actual)
    np.testing.assert_allclose(env[:, :3],future[:,:3],atol=1e-6)
    np.testing.assert_allclose(Rotation.from_quat(env[:,[4,5,6,3]]).as_matrix(),Rotation.from_quat(future[:,3:7]).as_matrix(),atol=1e-6)
    assert ds[3]['robotwin_metadata']['target_indices'][-1]==19
    with pytest.raises(IndexError): ds[4]
    ds.close()


def test_statistics_images_workers_and_source_immutability(tmp_path):
    ds = RobotwinHDF5Dataset(source(tmp_path),return_metadata=True)
    actions,states = [],[]
    for i in range(len(ds)):
        s,a = ds.read_lowdim(i)
        states.append(s)
        actions.append(actions_to_current_frame(s,a))
    for key,values in [('state',np.concatenate(states)),('action',np.concatenate(actions))]:
        np.testing.assert_allclose(ds.normalizer.params[key][0],values.astype(np.float64).mean(0),atol=1e-6)
        variance = values.astype(np.float64).std(0)
        np.testing.assert_allclose(ds.normalizer.params[key][1],np.where(variance<1e-6,1,variance),atol=1e-6)
    sample = ds[0]
    assert sample['state'].shape==(1,16) and sample['action'].shape==(16,16)
    assert all(im.size==(224,224) for im in sample['image'])
    assert np.asarray(sample['image'][0])[0,0,0]>200  # correct red, not blue
    loader = DataLoader(ds,batch_size=2,num_workers=2,multiprocessing_context='spawn',collate_fn=collate_fn)
    assert sum(len(batch) for batch in loader)==4
    ds.close()
    path = ds.root/ds.records[0]['path']
    with h5py.File(path,'a') as f: f.attrs['mutated']=1
    with pytest.raises(ValueError,match='Source changed'): ds[0]


def test_reject_invalid_pose():
    x = {'left_endpose':np.zeros(7),'right_endpose':np.zeros(7),'left_gripper':0.,'right_gripper':0.}
    with pytest.raises(ValueError,match='Zero'): poses_from_observation(x)


def test_eef_client_keeps_chunk_anchor(monkeypatch):
    from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_eef_interface as interface
    from starVLA.dataloader.robotwin_hdf5 import contract
    class FakeClient:
        calls = 0
        def __init__(self,*args): pass
        def get_server_metadata(self):
            c = dict(contract(), image_size=[24,32], include_state=True,
                     state_normalization={'mean':[0.]*16,'std':[1.]*16})
            return dict(action_contract=c, action_representation='current_eef_relative',action_chunk_size=16)
        def predict_action(self,payload):
            self.calls += 1
            assert payload['examples'][0]['state'].shape==(1,16)
            a=np.zeros((1,16,16),np.float32)
            a[...,0]=.01; a[...,6]=1; a[...,14]=1; a[...,7]=.5; a[...,15]=.5
            return {'data':{'actions':a}}
    monkeypatch.setattr(interface,'WebsocketClientPolicy',FakeClient)
    client=interface.ModelClient()
    observation={'endpose':{'left_endpose':[0,0,0,1,0,0,0], 'right_endpose':[0,0,0,1,0,0,0],
                           'left_gripper':.5,'right_gripper':.5},
                 'observation':{c:{'rgb':np.zeros((24,32,3),np.uint8)} for c in interface.CAMERAS}}
    first=client.step(observation,'task')
    observation['endpose']['left_endpose'][0]=10
    for i in range(15):
        np.testing.assert_allclose(client.step(observation,'task'),first)
    assert client.client.calls==1
    assert client.step(observation,'task')[0]>10
    assert client.client.calls==2
