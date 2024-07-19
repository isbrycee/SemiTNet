export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

/root/paddlejob/workspace/env_run/output/haojing/anaconda3_tooth_semi/bin/python3 -W ignore train_net.py \
        --config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
        --num-gpus 8 \
        --num-machines 1 \
        MODEL.WEIGHTS /root/paddlejob/workspace/env_run/output/haojing/SemiTNet/pretrained_model/SemiTNet_best_box_ap50.pth  \
        SSL.TRAIN_SSL False \
        OUTPUT_DIR MICCAI2024_output_allResize_lr1e-4_bs16/train_1st_teacher