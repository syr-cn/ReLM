{
name="run_Imidazo";
device=0;

python -W ignore main.py\
    --device $device --batch_size 256 \
    --model vicuna \
    --dataset ord --mode ord_dataset-10b940e7982c4622b1e1ac879394aba6_iupac \
    --corpus ord_dataset-10b9_5481_iupac \
    --task evaluate_llm --llm vicuna --max_tokens 200 \
    --k 4 --llm_bs 3 --num_sim 3 --gpt_temp 0.0\
    --smiles_wrap iupac_smiles --context uspto_train_small_iupac_5k \
    --openai_key "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\
    2>err/$name.err >log/$name.out;
}