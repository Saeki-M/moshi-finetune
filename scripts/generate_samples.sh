uv run accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    generate.py \
        --launcher accelerate \
        --output_dir output/20260401_1950-data_balanced/step_18500_fp32/continuation \
        --model_dir output/20260401_1950-data_balanced/step_18500_fp32 \
        --model_dtype float16 \
        --model_user_stream \
        --eval_data_files "moshi_finetune_data/parquet/llmjp_zoom1_train_001.parquet" \
        --num_examples 10 \
        --prompt_length 125 \
        --generation_length 250 \
        --temperature 0.8


uv run -m tools.detokenize_audio \
    --input output/20260401_1950-data_balanced/step_18500_fp32/continuation/generated_tokens \
    --output_dir output/20260401_1950-data_balanced/step_18500_fp32/continuation/generated_wavs \
    --output_format mp3