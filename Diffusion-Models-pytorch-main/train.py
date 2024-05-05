from types import SimpleNamespace


from ddpm_conditional import Diffusion


# This is a very simple example, for more advanced training, see `ddp_conditional.py`
# text_clip_1 = text -> 256, class -> 256, text + class
# text_clip_2 = text -> 256, class -> 256, cat(text,cat) -> 512, cat -> 256
config = SimpleNamespace(    
    run_name = "cifar_2",
    epochs = 300,
    noise_steps=1000,
    seed = 42,
    batch_size = 72,
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
    num_classes=10,
    lr = 5e-3)

diff = Diffusion(noise_steps=1000, img_size1=32, img_size2=32, num_class=10)

diff.prepare(config)
diff.fit(config)