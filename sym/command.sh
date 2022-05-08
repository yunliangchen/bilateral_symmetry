source venv/bin/activate

# Training (Optional)
# Execute the following commands to train the neural networks from scratch with four GPUs (specified by -d 0,1,2,3):

python ./train.py -d 2 --identifier depth_enabled_2 logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/config.yaml --from logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/checkpoint_latest.pth.tar
python ./train.py -d 1 --identifier test logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/config.yaml --from logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/checkpoint_latest.pth.tar
python ./train.py -d 0,1,2,3 --identifier baseline config/pix3d.yaml

# evaluate the models with coarse-to-fine inference for symmetry plane prediction and depth map estimation
# shapenet: ShapeNet/Symmetry
python eval_real.py -d 3 --output results/nerd.npz logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/config.yaml logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/checkpoint_latest.pth.tar
# shapenet-finetune: ShapeNet/Depth
python eval_real.py -d 1 --output results/nerd.npz logs/200513-030330-c8e671c-shapenet-finetune/config.yaml logs/220501-184703-5ac13d7-depth_disabled/checkpoint_best.pth
# shapenet, with depth as input - without depth supervision
python eval_real.py -d 2 --output results/nerd.npz logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/config.yaml logs/220501-184703-5ac13d7-depth_disabled/checkpoint_best.pth
# shapenet, with depth as input - with depth supervision
python eval_real.py -d 2 --output results/nerd.npz logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/config.yaml logs/220501-184712-5ac13d7-with_depth_enabled/checkpoint_best.pth


# real data
python eval_real.py -d 1 --output results/nerd.npz logs/real_config.yaml logs/200610-234002-8ee0ad2-shapenet/200610-234002-8ee0ad2-SN2-long-dropout/checkpoint_latest.pth.tar
# real data, shapenet-finetune: 
python eval_real.py -d 1 --output results/nerd.npz logs/real_config_shapenet_finetune.yaml logs/200513-030330-c8e671c-shapenet-finetune/checkpoint_best.pth.tar
# real data, with depth as input - without depth supervision
python eval_real.py -d 2 --output results/nerd.npz logs/real_config.yaml logs/220501-184703-5ac13d7-depth_disabled/checkpoint_best.pth
# real data, with depth as input - with depth supervision
python eval_real.py -d 2 --output results/nerd.npz logs/real_config.yaml logs/220501-184712-5ac13d7-with_depth_enabled/checkpoint_best.pth
