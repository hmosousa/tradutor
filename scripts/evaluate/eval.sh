# baselines
python scripts/evaluate/baselines.py -m deepl -d frmt
python scripts/evaluate/baselines.py -m google_br -d frmt
python scripts/evaluate/baselines.py -m google_pt -d frmt
python scripts/evaluate/baselines.py -m argo -d frmt
python scripts/evaluate/baselines.py -m deepl -d ntrex
python scripts/evaluate/baselines.py -m google_br -d ntrex
python scripts/evaluate/baselines.py -m google_pt -d ntrex
python scripts/evaluate/baselines.py -m argo -d ntrex
python scripts/evaluate/baselines.py -m opus_mt -d frmt
python scripts/evaluate/baselines.py -m opus_mt -d ntrex

# prompt
python scripts/evaluate/greedy.py -m google/gemma-2b-it -d frmt
python scripts/evaluate/greedy.py -m microsoft/Phi-3-mini-4k-instruct -d frmt
python scripts/evaluate/greedy.py -m meta-llama/Meta-Llama-3-8B-Instruct -d frmt
python scripts/evaluate/greedy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct -d frmt
python scripts/evaluate/greedy.py -m google/gemma-2b-it -d ntrex
python scripts/evaluate/greedy.py -m microsoft/Phi-3-mini-4k-instruct -d ntrex
python scripts/evaluate/greedy.py -m meta-llama/Meta-Llama-3-8B-Instruct -d ntrex
python scripts/evaluate/greedy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct -d ntrex

# lora
python scripts/evaluate/greedy.py -m u1537782/hf_gemma_lora -d frmt
python scripts/evaluate/baselines.py -m phi3_lora -c phi3_lora -d frmt
python scripts/evaluate/baselines.py -m llama3_lora -c llama3_lora -d frmt
python scripts/evaluate/greedy.py -m hugosousa/hf_gemma_lora -d ntrex
python scripts/evaluate/baselines.py -m phi3_lora -c phi3_lora -d ntrex
python scripts/evaluate/baselines.py -m llama3_lora -c llama3_lora -d ntrex

# fft
python scripts/evaluate/greedy.py -m u1537782/hf_gemma_fft -d frmt
python scripts/evaluate/greedy.py -m u1537782/hf_phi3_fft_rd -d frmt
python scripts/evaluate/greedy.py -m u1537782/hf_llama3_fft -d frmt
python scripts/evaluate/greedy.py -m u1537782/llama3.1_fft -d frmt
python scripts/evaluate/greedy.py -m u1537782/hf_gemma_fft -d ntrex
python scripts/evaluate/greedy.py -m u1537782/hf_phi3_fft_rd -d ntrex
python scripts/evaluate/greedy.py -m u1537782/hf_llama3_fft -d ntrex
python scripts/evaluate/greedy.py -m u1537782/llama3.1_fft -d ntrex

# print the results

echo "FRMT"
python scripts/print_results.py -d frmt

echo "NTREX"
python scripts/print_results.py -d ntrex
