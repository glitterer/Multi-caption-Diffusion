/var/lib/slurm/slurmd/job10176/slurm_script: line 15: cd: /home/soh62/Multi-caption-Diffusion/Diffusion-Models-pytorch-main: No such file or directory
Traceback (most recent call last):
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/train.py", line 4, in <module>
    from ddpm_conditional import Diffusion
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/ddpm_conditional.py", line 20, in <module>
    from Embedding import clip_text_embedding, clip_image_embedding, t5_embedding
  File "/ifs/CS/replicated/home/soh62/DLProject/Multi-caption-Diffusion/Diffusion-Models-pytorch-main/Embedding.py", line 3, in <module>
    from transformers import AutoTokenizer, T5EncoderModel
ModuleNotFoundError: No module named 'transformers'
