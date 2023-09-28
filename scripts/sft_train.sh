datasets=('shp' 'hh' 'jeopardy')
seeds=(0 1 2 3 4)
for dataset in "${datasets[@]}"
do
    python -u train_qlora.py model=llama7b datasets=["$dataset"] loss=sft exp_name="$dataset"_sft gradient_accumulation_steps=4 batch_size=32 eval_batch_size=32 sample_during_eval=True have_llm_dropout=True seed=0
done