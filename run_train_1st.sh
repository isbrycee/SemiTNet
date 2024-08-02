export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -W ignore train_net.py \
        --config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
        --num-gpus 8 \
        --num-machines 1 \
        MODEL.WEIGHTS ./pretrained_model/SemiTNet_best_box_ap50.pth  \
        SSL.TRAIN_SSL False \
        OUTPUT_DIR output/train_1st_teacher
