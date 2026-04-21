import time
import numpy as np
import ctranslate2
import transformers
import sacrebleu
import requests
import os

# --- 1. CONFIGURATION & SETUP ---
MODEL_ID = "facebook/nllb-200-distilled-600M"
CT2_MODEL_DIR = "nllb_ct2"
DEVICE = "cuda" # Use "cpu" if no GPU is available

def setup_environment():
    print("Checking environment...")
    # Convert model to CTranslate2 format if it doesn't exist
    if not os.path.exists(CT2_MODEL_DIR):
        print(f"Converting {MODEL_ID} to CTranslate2 INT8...")
        # Command line conversion call
        os.system(f"ct2-transformers-converter --model {MODEL_ID} --output_dir {CT2_MODEL_DIR} --quantization int8_float16")

def download_flores_raw(lang_code):
    url = f"https://raw.githubusercontent.com/facebookresearch/flores/main/flores200_dataset/dev/{lang_code}.dev"
    return requests.get(url).text.strip().split('\n')

# --- 2. THE BENCHMARK ENGINE ---
def run_benchmark():
    setup_environment()
    
    # Load Data
    print("Fetching FLORES-200 evaluation data...")
    inputs = download_flores_raw("eng_Latn")[:100]
    targets = download_flores_raw("spa_Latn")[:100]

    # Load Model
    translator = ctranslate2.Translator(CT2_MODEL_DIR, device=DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    latencies = []
    results = []
    
    # --- WARM UP ---
    print("Warming up engine...")
    warmup_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("Warmup sentence for GPU kernels."))
    for _ in range(5):
        _ = translator.translate_batch([warmup_tokens], target_prefix=[["spa_Latn"]])

    # --- MEASUREMENT ---
    print(f"Executing benchmark on {len(inputs)} samples...")
    for text in inputs:
        start = time.perf_counter()
        
        # Fast Tokenization
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        
        # Translation with Production Settings (Beam Size 1 = Max Speed)
        output = translator.translate_batch(
            [tokens], 
            target_prefix=[["spa_Latn"]],
            beam_size=1, 
            max_decoding_length=256
        )
        
        # Decode
        translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(output[0].hypotheses[0]))
        
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(translation)
        
    # --- METRICS ---
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    bleu = sacrebleu.corpus_bleu(results, [targets]).score
    
    print("\n" + "="*30)
    print("STELLA TRANSLATION BENCHMARK")
    print("="*30)
    print(f"Model: {MODEL_ID} (INT8)")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms")
    print(f"BLEU Score:  {bleu:.2f}")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
