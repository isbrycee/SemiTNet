export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -W ignore train_net.py \
        --config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
        --eval-only \
        --num-gpus 8 \
        --num-machines 1 \
        MODEL.WEIGHTS model_0004999.pth \
        SSL.PERCENTAGE 10 \
        SSL.TRAIN_SSL True \
        SSL.EVAL_WHO STUDENT \
        SSL.TEACHER_CKPT model_0002499.pth \
       #  OUTPUT_DIR output/train_1st_teacher