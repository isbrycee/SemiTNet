export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -W ignore train_net.py \
    --config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
    --num-gpus 8 --num-machines 1 \
    MODEL.WEIGHTS ./pretrained_models/mobile_sam.pkl \
    SSL.PERCENTAGE 100 \
    SSL.TRAIN_SSL True \
    SSL.TEACHER_CKPT output/train_1st_teacher/model_best.pth \
    OUTPUT_DIR output/train_2st_student_allBurnIn/ \
    SSL.BURNIN_ITER 26250 \
    SSL.EVAL_WHO STUDENT \
