/var/lib/slurm/slurmd/job10190/slurm_script: line 14: activate: No such file or directory
/var/lib/slurm/slurmd/job10190/slurm_script: line 15: cd: /home/soh62/Multi-caption-Diffusion/Diffusion-Models-pytorch-main: No such file or directory
05:11:01 - INFO: Starting epoch 0:
Traceback (most recent call last):
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/train.py", line 30, in <module>
    diff.fit(config)
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/ddpm_conditional.py", line 181, in fit
    _  = self.one_epoch(train=True)
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/ddpm_conditional.py", line 135, in one_epoch
    predicted_noise = self.model(x_t, t, labels)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/modules.py", line 213, in forward
    return self.unet_forwad(x, t)
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/modules.py", line 191, in unet_forwad
    x = self.sa6(x)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/modules.py", line 56, in forward
    attention_value, _ = self.mha(x_ln, x_ln, x_ln)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/modules/activation.py", line 1266, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/functional.py", line 5470, in multi_head_attention_forward
    attn_output_weights = softmax(attn_output_weights, dim=-1)
  File "/home/soh62/miniconda3/envs/DLproject/lib/python3.10/site-packages/torch/nn/functional.py", line 1885, in softmax
    ret = input.softmax(dim)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB. GPU 
