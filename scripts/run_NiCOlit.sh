{
name="run_NiCOlit";
device=0;

python main.py\
    --device $device --batch_size 256 \
    --model vicuna \
    --dataset ord --mode ord_dataset-9b8aa9a7835143ef8ce3f70abfab7545_iupac \
    --corpus ord_dataset-9b8aa9a7835143ef8ce3f70abfab7545_iupac \
    --task evaluate_llm --llm vicuna --max_tokens 200 \
    --k 4 --llm_bs 3 --num_sim 3 --gpt_temp 0.0\
    --smiles_wrap iupac_smiles --context uspto_train_iupac_5k \
    --openai_key "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"\
    2>err/$name.err >log/$name.out;
}