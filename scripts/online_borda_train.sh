#python -u train_qlora.py model=llama7b datasets=[jeopardy] loss=dpo loss.beta=0.1 model.archive=TODO exp_name=jeopardy_active_llama7b_dropout gradient_accumulation_steps=6 batch_size=32 eval_batch_size=32  sample_during_eval=True active=True pretrain=False have_llm_dropout=True
# python -u train_qlora.py model=llama7b datasets=[hh] loss=dpo loss.beta=0.1 model.archive=TODO exp_name=hh_active_llama7b_dropout gradient_accumulation_steps=6 batch_size=32 eval_batch_size=32  sample_during_eval=True active=True pretrain=False have_llm_dropout=True
python -u train_qlora.py model=phi2 datasets=[jokes] loss=sft exp_name=test gradient_accumulation_steps=2 batch_size=32 eval_batch_size=8 sample_during_eval=True active=False pretrain=True have_llm_dropout=True debug=True
# python -u train_qlora.py model=phi2 datasets=[jokes] loss=dpo loss.beta=0.1 exp_name=test gradient_accumulation_steps=6 batch_size=32 eval_batch_size=8  sample_during_eval=True online=True pretrain=False have_llm_dropout=True debug=True do_first_eval=False
#
#python -u train_qlora.py model=llama7b datasets=[shp] loss=dpo loss.beta=0.1 model.archive=/home/scratch/vdas/dpo/shp_epillama_1.0_2023-08-23_21-54-24_330522/LATEST/policy.pt exp_name=shp_active_llama7b_dropout gradient_accumulation_steps=6 batch_size=32 eval_batch_size=32  sample_during_eval=True active=True pretrain=False epinet=True

