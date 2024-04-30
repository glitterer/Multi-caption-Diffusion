from types import SimpleNamespace


from ddpm_conditional import Diffusion


# This is a very simple example, for more advanced training, see `ddp_conditional.py`

config = SimpleNamespace(    
    run_name = "text_con_diff",
    epochs = 40,
    noise_steps=1000,
    seed = 42,
    batch_size = 28,
    img_size = 64,
    num_classes = 10,
    train_folder = "train",
    val_folder = "test",
    device = "cuda:1",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3)

diff = Diffusion(noise_steps=config.noise_steps , img_size1=config.img_size, img_size2=config.img_size)

diff.prepare(config)
diff.fit(config)