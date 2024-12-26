
import subprocess
import os
subprocess.call('pip3 install torch torchvision torchaudio', shell=True)
import torch

torch_version, cuda_version = torch.__version__.split('+')
file_name = 'mmstart.sh'
with open(file_name, 'w') as f:
    f.write(
        f'pip3 install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/torch{torch_version}/index.html')
subprocess.call('bash mmstart.sh', shell=True)
os.remove(file_name)
subprocess.call('pip3 install -r requirements.txt', shell=True)
subprocess.call('bash mmdetection_install.sh', shell=True)