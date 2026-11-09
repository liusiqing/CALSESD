"""先训练共享模型；不声称 rollout 后端参与训练。参数转交原 train_ppo。"""
import subprocess
import sys
from pathlib import Path
if __name__=='__main__':
    script=Path(__file__).resolve().parents[1]/'training'/'train_ppo.py'
    subprocess.run([sys.executable,str(script),*sys.argv[1:]],check=True)
