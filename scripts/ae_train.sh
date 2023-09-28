datasets=('shp' 'hh' 'jeopardy')
seeds=(0 1 2 3 4)
for dataset in "${datasets[@]}"
do
  for seed in "${seeds[@]}"
  do
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1536" python -u train_qlora.py model=llama7b datasets=["$dataset"] loss=dpo loss.beta=0.1 active_beta=4.0 model.archive=<SFT_PATH> exp_name="$dataset"_active gradient_accumulation_steps=4 batch_size=32 eval_batch_size=32 sample_during_eval=True active=True pretrain=False have_llm_dropout=True max_train_examples=30000 seed="$seed"
  done
done