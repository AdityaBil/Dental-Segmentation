import os
import sys
import time
import glob
import subprocess
out = []
try:
    out.append(f'cwd={os.getcwd()}')
    out.append(f'python={sys.executable}')
    out.append(f'python version={sys.version}')
    out.append('exists checkpt=' + str(os.path.exists('checkpoints')))
    if os.path.exists('checkpoints'):
        for p in sorted(glob.glob(os.path.join('checkpoints','*'))):
            out.append(f'{p} | mtime={time.ctime(os.path.getmtime(p))} | size={os.path.getsize(p)}')
    try:
        import torch
        out.append('torch import ok')
        if os.path.exists('checkpoints/best_model.pth'):
            ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
            out.append('epoch=' + str(ckpt.get('epoch')))
            if 'enc1.block.0.weight' in ckpt['state_dict']:
                out.append('enc1 shape=' + str(tuple(ckpt['state_dict']['enc1.block.0.weight'].shape)))
    except Exception as e:
        out.append('torch error: ' + repr(e))
except Exception as e:
    out.append('outer error: ' + repr(e))
with open('shell_debug_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
