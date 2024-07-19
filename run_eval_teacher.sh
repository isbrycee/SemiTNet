python3 -W ignore train_net.py \
        --config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
        --eval-only \
        --num-gpus 8 \
        --num-machines 1 \
        MODEL.WEIGHTS output_allResize_lr1e-4_bs16/train_1st_teacher/model_best.pth \
        SSL.PERCENTAGE 10 \
        SSL.TRAIN_SSL False \
        SSL.EVAL_WHO Teacher \
       # SSL.TEACHER_CKPT model_best.pth \
       #  OUTPUT_DIR output/train_1st_teacher