export CUDA_VISIBLE_DEVICES=0

NOW=$(date +'%Y_%m_%d__%H_%M_%Z')
MODEL_ARCHITECTURE=CNNwithBN # CNN, CNNwithBN, MobileNetV2, MobileNetV3
OUTPUT_DIR=./resources/outputs/$MODEL_ARCHITECTURE/$NOW

python -m trainer.main \
    --output_dir $OUTPUT_DIR \
    --model_architecture $MODEL_ARCHITECTURE \
    --lr 0.001 \
    --max_epochs 100 \
    --conv_type separable \
    --loss_fct nll \
