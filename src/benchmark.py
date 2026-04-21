import time
import numpy as np
import ctranslate2
import transformers
import sacrebleu
import requests
import os

# --- 1. CONFIGURATION ---
MODEL_ID = "facebook/nllb-200-distilled-600M"
CT2_MODEL_DIR = "nllb_ct2"
DEVICE = "cuda"  # Ensure Colab is set to T4 GPU

def setup_environment():
    print("Checking environment and converting model...")
    if not os.path.exists(CT2_MODEL_DIR):
        # Convert to INT8 for max speed and low memory footprint
        os.system(f"ct2-transformers-converter --model {MODEL_ID} --output_dir {CT2_MODEL_DIR} --quantization int8_float16")

def download_flores_raw(lang_code):
    url = f"https://raw.githubusercontent.com/facebookresearch/flores/main/flores200_dataset/dev/{lang_code}.dev"
    return requests.get(url).text.strip().split('\n')

# --- 2. EVALUATION SUITE ---

def run_medical_validation(translator, tokenizer):
    print("\n--- MEDICAL DOMAIN SANITY CHECK ---")
    medical_checks = [
        "The patient requires a blood pressure check.",
        "Please take two tablets after every meal.",
        "The MRI results show no signs of fracture.",
        "Are you experiencing any shortness of breath?"
    ]
    for text in medical_checks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        output = translator.translate_batch([tokens], target_prefix=[["spa_Latn"]], beam_size=1)
        translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(output[0].hypotheses[0]))
        print(f"EN: {text}")
        print(f"ES: {translation}")

def benchmark_batch_throughput(translator, tokenizer):
    print("\n--- SCALABILITY: BATCH THROUGHPUT SIMULATION ---")
    # Simulate a burst of 16 concurrent requests
    batch_text = ["This is a test sentence for concurrent processing."] * 16
    batch_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in batch_text]
    
    start = time.perf_counter()
    _ = translator.translate_batch(batch_tokens, target_prefix=[["spa_Latn"]]*16, beam_size=1)
    end = time.perf_counter()
    
    total_time_ms = (end - start) * 1000
    print(f"Processed Batch of 16 sentences in: {total_time_ms:.2f} ms")
    print(f"Effective per-request latency: {total_time_ms/16:.2f} ms")

def run_main_benchmark():
    setup_environment()
    
    # Load Data (100 samples for statistical validity)
    print("\nFetching 100 samples from FLORES-200...")
    inputs = download_flores_raw("eng_Latn")[:100]
    targets = download_flores_raw("spa_Latn")[:100]

    # Load Model
    translator = ctranslate2.Translator(CT2_MODEL_DIR, device=DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    latencies = []
    results = []
    
    # --- WARM UP ---
    warmup_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("Warmup."))
    for _ in range(5):
        _ = translator.translate_batch([warmup_tokens], target_prefix=[["spa_Latn"]])

    # --- CORE BENCHMARK ---
    print(f"Executing benchmark...")
    for text in inputs:
        start = time.perf_counter()
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        output = translator.translate_batch([tokens], target_prefix=[["spa_Latn"]], beam_size=1)
        translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(output[0].hypotheses[0]))
        
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(translation)
        
    # --- METRICS ---
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    bleu = sacrebleu.corpus_bleu(results, [targets]).score
    
    print("\n" + "="*40)
    print("FINAL STELLA BENCHMARK RESULTS")
    print("="*40)
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms (Target: <150ms)")
    print(f"BLEU Score:  {bleu:.2f} (Target: >85)")
    print("="*40)

    # Run Additional Validations
    run_medical_validation(translator, tokenizer)
    benchmark_batch_throughput(translator, tokenizer)

if __name__ == "__main__":
    run_main_benchmark()
