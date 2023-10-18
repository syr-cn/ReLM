{
name="run_Rexgen40k";
device=0;

python -W ignore main.py\
    --device $device --batch_size 256 \
    --model vicuna \
    --dataset ord --mode ord_dataset-488402f6ec0d441ca2f7d6fabea7c220_iupac \
    --corpus ord_dataset-488402f6ec0d441ca2f7d6fabea7c220_iupac \
    --task evaluate_llm --llm vicuna --max_tokens 200 \
    --k 4 --llm_bs 3 --num_sim 3 --gpt_temp 0.0\
    --smiles_wrap iupac_smiles --context uspto_train_iupac_5k \
    --openai_key "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\
    2>err/$name.err >log/$name.out;
}