from types import SimpleNamespace


from ddpm_conditional import Diffusion


# This is a very simple example, for more advanced training, see `ddp_conditional.py`

config = SimpleNamespace(    
    run_name = "text_con_ddpm",
    epochs = 40,
    noise_steps=1000,
    seed = 42,
    batch_size = 32,
    img_size = 32,
    text_embed_length = 256,
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3)

diff = Diffusion(noise_steps=config.noise_steps , img_size1=config.img_size, img_size2=config.img_size, num_class=10)

diff.load('/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/test', 'checkpt_class.pt', 'ema_checkpt_class.pt')
diff.log_images(160)

# diff.load('/mnt/c/Users/rdeme/Documents/Brown/CSCI_2470_Deep_Learning/project/test', 'checkpt_2.pt', 'ema_checkpt_2.pt')
# diff.generate_fid_score()
