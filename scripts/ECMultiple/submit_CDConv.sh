cd ../../examples_3D

export model_3d=CDConv

export lr_list=(5e-4 1e-4 1e-3)
export seed=42
export batch_size_list=(8 16 64)


export epochs=500
export dataset=ECMultiple
export StepLRCustomized_scheduler="60"


export lr_scheduler=CosineAnnealingLR

for lr in "${lr_list[@]}"; do

for batch_size in "${batch_size_list[@]}"; do

    export output_model_dir=../output/"$model_3d"/"$dataset"/"$seed"/"$lr"_"$lr_scheduler"_"$batch_size"_"$epochs"
    export output_file="$output_model_dir"/result.out
    echo "$output_model_dir"
    mkdir -p "$output_model_dir"

    python finetune_ECMultiple.py \
    --model_3d="$model_3d" --dataset="$dataset" --epochs="$epochs" \
    --seed="$seed" \
    --batch_size="$batch_size" --optimizer SGD --CDConv_kernel_channels 32 --CDConv_base_width 16 \
    --lr="$lr" --lr_scheduler="$lr_scheduler" --print_every_epoch=1 \
    --output_model_dir="$output_model_dir" \
    > "$output_file"

done
done
