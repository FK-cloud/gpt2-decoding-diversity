# config for the experiment - change stuff here, rest of the code just imports from this

MODEL_NAME = "gpt2"          # GPT-2 Small (124M)
TOP_P = 0.9
TOPP_SEEDS = [11, 22, 33]     # 3 seeds per prompt for top-p
GEN_LENGTH = 200              # how many tokens to generate each run
ANALYSIS_LENGTHS = [50, 100, 200]     # we just truncate the same generation at these lengths
N_BOOTSTRAP = 5000          
CI = 0.95
RNG_SEED_BOOTSTRAP = 2026     # so the bootstrap numbers don't change every time I rerun this

RESULTS_DIR = "results"
