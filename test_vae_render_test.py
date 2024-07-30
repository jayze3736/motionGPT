import os
import sys
os.system('pip install /home/user/app/pyrender')
os.system('pip install pyglet==1.4.0a1')
os.system('pip install triangle==20220202')
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
sys.path.append("/data/motionGPT")
import gradio as gr
import torch
import time
import numpy as np
import pytorch_lightning as pl
import subprocess
from pathlib import Path
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from huggingface_hub import snapshot_download
import json
import argparse

# def main_parse_args():
#     desc = "rendering input params"
#     parser = argparse.ArgumentParser(description=desc)
#     cfg = parse_args(phase="render")  # parse config file
#     print("cfg")
#     print(cfg)
#     # GPU 번호
#     parser.add_argument(
#         "--save_dir", type=str, required=False, default='/data/motionGPT/workspace/vae_exp'
#     )
#     parser.add_argument(
#         "--json_path", type=str, required=True
#     )
#     parser.add_argument(
#         "--gen_mode", type=str, required=False, default='default', choices=['default', 'reverse' ,'random']
#     )
#     args = parser.parse_args()
#     print("args")
#     print(args)
#     exit()
#     return cfg

# Load model
cfg = parse_args(phase="render")  # parse config file
cfg.FOLDER = 'test_visualization'

output_dir = Path(cfg.FOLDER)
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model_path = snapshot_download(repo_id="bill-jiang/MotionGPT-base")

datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule)
state_dict = torch.load(f'{model_path}/motiongpt_s3_h3d.tar',
                        map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.to(device)
# 
model.lm.device = device
## 

# audio_processor = WhisperProcessor.from_pretrained(cfg.model.whisper_path)
# audio_model = WhisperForConditionalGeneration.from_pretrained(
#     cfg.model.whisper_path).to(device)
# forced_decoder_ids_zh = audio_processor.get_decoder_prompt_ids(
#     language="zh", task="translate")
# forced_decoder_ids_en = audio_processor.get_decoder_prompt_ids(
#     language="en", task="translate")


# motion_length, motion_token_string = motion_uploaded[
#         "motion_lengths"], motion_uploaded["motion_token_string"]

#     input = data_stored[-1]['user_input']
#     prompt = model.lm.placeholder_fulfill(input, motion_length,
#                                           motion_token_string, "")
#     data_stored[-1]['model_input'] = prompt
#     batch = {
#         "length": [motion_length],
#         "text": [prompt],
#     }

#     outputs = model(batch, task="t2m")
#     out_feats = outputs["feats"][0]
#     out_lengths = outputs["length"][0]
#     out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
#     out_texts = outputs["texts"][0]

# np file -> rendered video
def load_motion(motion_uploaded):
    file = motion_uploaded['file']

    
    
    feats = torch.tensor(np.load(file), device=model.device)

    if len(feats.shape) == 2:
        feats = feats[None]
    # feats = model.datamodule.normalize(feats)

    # Motion tokens
    print("Features")
    print(feats)
    print("Shape")
    print(feats.shape)
    motion_lengths = feats.shape[0]
    motion_token, _ = model.vae.encode(feats)

    motion_token_string = model.lm.motion_token_to_string(
        motion_token, [motion_token.shape[1]])[0]
    motion_token_length = motion_token.shape[1]

    
    # Motion rendered
    
    joints = model.datamodule.feats2joints(feats.cpu()).cpu().numpy()
    # _, _, output_npy_path, joints_fname = render_motion(
    #     joints,
    #     feats.to('cpu').numpy(), method)

    motion_uploaded.update({
        "feats": feats,
        "joints": joints,
        "motion_video": None,
        "motion_video_fname": None,
        "motion_joints": None,
        "motion_joints_fname": None,
        "motion_lengths": motion_lengths,
        "motion_token": motion_token,
        "motion_token_string": motion_token_string,
        "motion_token_length": motion_token_length,
    })

    return motion_uploaded

def edit_motion(motion_token_string, mode='default'):

    # token sequence
    
    print(f"edit motion motion_token_string:{motion_token_string}")

    if mode=='default':
        pass
    elif mode=='random':

        import random
        # SOM, EOM token 모두 제거
        motion_token_string = motion_token_string.replace("<motion_id_512>", "")
        motion_token_string = motion_token_string.replace("<motion_id_513>", "")
        motion_tokens = motion_token_string.split('><')
        
        motion_tokens = [f'<{token.replace(">", "").replace("<","")}>' if ('<' in token or '>' in token) else f'<{token}>' for token in motion_tokens]

        # 랜덤으로 시퀀스 섞기
        random.shuffle(motion_tokens)

        # 섞인 시퀀스를 다시 문자열로 결합
        motion_token_string = ''.join(motion_tokens)
        prefix = "<motion_id_512>"
        suffix = "<motion_id_513>"
        motion_token_string = prefix + motion_token_string + suffix
        print(f"Shuffled motion token string: {motion_token_string}")
    elif mode=='reverse':

        # SOM, EOM token 모두 제거
        motion_token_string = motion_token_string.replace("<motion_id_512>", "")
        motion_token_string = motion_token_string.replace("<motion_id_513>", "")
        motion_tokens = motion_token_string.split('><')
        
        motion_tokens = [f'<{token.replace(">", "").replace("<","")}>' if ('<' in token or '>' in token) else f'<{token}>' for token in motion_tokens]

        # 랜덤으로 시퀀스 섞기
        motion_tokens = motion_tokens.reverse()

        # 섞인 시퀀스를 다시 문자열로 결합
        motion_token_string = ''.join(motion_tokens)
        prefix = "<motion_id_512>"
        suffix = "<motion_id_513>"
        motion_token_string = prefix + motion_token_string + suffix
        print(f"reversed motion token string: {motion_token_string}")

    # model.vae.decode
   
    motion_tokens, _ = model.lm.motion_string_to_token([motion_token_string])
    m_tokens = motion_tokens[0]
    feats = model.vae.decode(m_tokens)
    gen_feats = feats.to('cpu').numpy()
    #print(f"edit motion gen_feats:{feats}")
    gen_joints = model.feats2joints(feats).to('cpu').numpy()

        #print(f"edit motion gen_joints:{gen_joints}")

    return gen_feats, gen_joints


def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string

# render motion
"""
data: joint
feats: motion features
"""
def render_motion(data, feats, fname, method='fast'):
    # fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
    #    time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = f"{fname}_feats" + '.npy'
    data_fname = f"{fname}_joints" + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_joints_path = os.path.join(output_dir, data_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    np.save(output_npy_path, feats)
    np.save(output_joints_path, data)
    
    render_cmd = ["python", "-m", "render", "--joints_path", output_joints_path, "--method", method, "--output_mp4_path", output_mp4_path, "--smpl_model_path", cfg.RENDER.SMPL_MODEL_PATH]
    os.system(" ".join(render_cmd))
    # subprocess.run(cmd3)
    
    return output_mp4_path, video_fname, output_npy_path, feats_fname

def render_motion_with_only_joints(data, fname, method='fast'):
    # fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
    #    time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    data_fname = f"{fname}_joints" + '.npy'
    output_joints_path = os.path.join(output_dir, data_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    
    np.save(output_joints_path, data)
    
    render_cmd = ["python", "-m", "render", "--joints_path", output_joints_path, "--method", method, "--output_mp4_path", output_mp4_path, "--smpl_model_path", cfg.RENDER.SMPL_MODEL_PATH]
    os.system(" ".join(render_cmd))
    # subprocess.run(cmd3)
    
    return output_mp4_path, video_fname



# input: motion string token
# load vae model - decoder
# rendering 
# output: rendered video 

if __name__ == "__main__":

    print("Start!")
    
    save_dir = '/data/motionGPT/workspace/vae_exp'
    input_path = '/data/motionGPT/workspace/vae_exp/decoder_input/0_joints.json' # /data/motionGPT/workspace/vae_exp/encoder_input/n1.json'
    mode = 'default' # 'default', 'reverse', 'random'

    with open(input_path, 'r') as json_file:
        input = json.load(json_file)
    
    encode = input['encode']
    file = input['encode_motion_file']
    decode = input['decode']
    m_tokens = input['user_input_token_seq']
    file_name = file.split('/')[-1].replace('.npy', '')
    if encode:
        motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
        }
        motion_uploaded['file'] = file
        method = 'fast'
        motion_uploaded = load_motion(motion_uploaded=motion_uploaded) # 특정 모션에 대하여 load -> token sequence도 반환
        motion_token_string = motion_uploaded['motion_token_string']
        print(f"file_path:{file}")
        print(f"motion_token_string:{motion_uploaded['motion_token_string']}")
        print(f"motion_token_length:{motion_uploaded['motion_token_length']}")
        
        # json 파일 저장
        input['user_input_token_seq'] = motion_token_string
        input['decode'] = True
        input['encode'] = False
        
        with open(f'{save_dir}/decoder_input/{file_name}.json', 'w') as json_file:
            json.dump(input, json_file, indent=4)
        print("### Decoder input saved ###")
    elif decode:
        file_type = file.split('/')[-1].replace('.npy', '')
        if 'joints' in file_type:
            gen_joints = torch.tensor(np.load(file), device=model.device)
            gen_joints = gen_joints.cpu().numpy()
            render_motion_with_only_joints(data=gen_joints,fname=file_name,method='slow')
            print("### Rendered motion saved ###")
        else:
            motion_token_string = m_tokens
            fname = file_name + f'_mode_{mode}'
            gen_feats, gen_joints = edit_motion(motion_token_string, mode) # 원하는 의도에 따라서 토큰 시퀀스 변경 및 vae decode
            render_motion(data=gen_joints, fname = fname, feats=gen_feats, method='slow') # data를 rendering에 연결
            print("### Rendered motion saved ###")
    print("Done!")