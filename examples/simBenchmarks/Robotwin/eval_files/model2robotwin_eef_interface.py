"""UMI-relative EEF deployment. The anchor is fixed for the entire action chunk."""
import numpy as np
from PIL import Image
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.dataloader.robotwin_hdf5 import VERSION, CAMERAS, poses_from_observation, to_environment_actions


class ModelClient:
    def __init__(self, policy_ckpt_path=None, host='127.0.0.1', port=5694, unnorm_key='robotwin_hdf5'):
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        meta = self.client.get_server_metadata()
        self.contract = meta.get('action_contract', {})
        if self.contract.get('version') != VERSION or meta.get('action_representation') != 'current_eef_relative':
            raise ValueError('Expected a native RoboTwin UMI-relative EEF checkpoint')
        self.horizon = int(meta['action_chunk_size'])
        if self.horizon != 16 or self.contract.get('camera_order') != list(CAMERAS):
            raise ValueError('Checkpoint input/action contract differs from adapter')
        self.image_size = tuple(self.contract['image_size'])
        self.reset()

    def reset(self, task_description=None):
        self.actions, self.cursor, self.instruction, self.previous_state = None, 0, task_description, None

    def step(self, observation, instruction):
        if self.instruction != instruction:
            self.reset(instruction)
        observed = poses_from_observation(observation['endpose'])
        if self.previous_state is not None:
            for start in (3,11):
                if np.dot(observed[start:start+4], self.previous_state[start:start+4]) < 0:
                    observed[start:start+4] *= -1
        self.previous_state = observed.copy()
        if self.actions is None or self.cursor == len(self.actions):
            anchor = observed
            example = dict(lang=instruction, image=[
                np.asarray(Image.fromarray(observation['observation'][camera]['rgb']).resize(self.image_size))
                for camera in CAMERAS])
            if self.contract['include_state']:
                stats = self.contract['state_normalization']
                example['state'] = ((anchor-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32))[None]
            result = self.client.predict_action({'examples':[example], 'unnorm_key':self.unnorm_key})
            relative = np.asarray(result['data']['actions'][0],np.float32)
            if relative.shape != (16,16) or not np.isfinite(relative).all():
                raise ValueError('Expected finite [16,16] relative action chunk')
            # Inverse scaling happens at the server. Decode once using the original anchor.
            self.actions = to_environment_actions(anchor, relative)
            self.cursor = 0
        action = self.actions[self.cursor].copy()
        self.cursor += 1
        return action


def get_model(usr_args):
    return ModelClient(**{k:usr_args[k] for k in ('policy_ckpt_path','host','port','unnorm_key') if k in usr_args})


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    action = model.step(observation, str(TASK_ENV.get_instruction()))
    TASK_ENV.take_action(action, action_type='ee')
