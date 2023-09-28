datasets=('shp' 'hh' 'jeopardy')
seeds=(0 1 2 3 4)
for dataset in "${datasets[@]}"
do
  for seed in "${seeds[@]}"
  do
    python -u train_qlora.py model=llama7b datasets=["$dataset"] loss=dpo loss.beta=0.1 model.archive=<SFT_PATH> exp_name="$dataset"_dpo gradient_accumulation_steps=4 batch_size=32 eval_batch_size=32 sample_during_eval=True active=False pretrain=False have_llm_dropout=True max_train_examples=30000 seed="$seed"
  done
done