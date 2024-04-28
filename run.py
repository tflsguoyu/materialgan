import os

cmd = 'python src/optim.py' \
    + ' --in_dir data/in/' \
    + ' --mat_fn real_cards-blue' \
    + ' --out_dir data/out' \
    + ' --vgg_weight_dir data/pretrain/vgg_conv.pt' \
    + ' --num_render_used 9' \
    + ' --epochs 2000' \
    + ' --sub_epochs 10 10' \
    + ' --loss_weight 1000 0.001 -1 -1'\
    + ' --optim_latent' \
    + ' --lr 0.02' \
    + ' --gan_latent_init data/pretrain/latent_avg_W+_256.pt' \

os.system(cmd)
