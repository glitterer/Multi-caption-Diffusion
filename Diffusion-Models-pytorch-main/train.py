from types import SimpleNamespace


from ddpm_conditional import Diffusion


# This is a very simple example, for more advanced training, see `ddp_conditional.py`
# text_clip_1 = text -> 256, class -> 256, text + class
# text_clip_2 = text -> 256, class -> 256, cat(text,cat) -> 512, cat -> 256
config = SimpleNamespace(    
    run_name = "text_clip_2_diff",
    epochs = 80,
    noise_steps=1000,
    seed = 42,
    batch_size = 64,
    img_size = 32,
    num_classes = 10,
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3)

diff = Diffusion(noise_steps=config.noise_steps , img_size1=config.img_size, img_size2=config.img_size, num_class=config.num_classes)

diff.prepare(config)
diff.fit(config)