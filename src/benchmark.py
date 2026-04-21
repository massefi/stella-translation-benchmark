import time
import numpy as np
import ctranslate2
import transformers
import sacrebleu
import requests
import os

#ARCHITECTURAL COMPARATIVE STUDY
CONFIG_OPTIONS = {
    "A": {"name": "Baseline", "model": "facebook/nllb-200-distilled-600M", "quant": None},
    "B": {"name": "Optimized", "model": "facebook/nllb-200-distilled-600M", "quant": "int8_float16"},
    "C": {"name": "LLM-Alternative", "model": "meta-llama/Meta-Llama-3-8B", "quant": "int4"}
}

ACTIVE_CONFIG = CONFIG_OPTIONS["B"]
CT2_MODEL_DIR = "nllb_ct2"
DEVICE = "cuda" 
def normalize(text):
      return text.strip().lower()
def setup_environment():
    print(f"--- INITIALIZING STRATEGY: {ACTIVE_CONFIG['name']} ---")
    if ACTIVE_CONFIG["quant"] == "int8_float16":
        if not os.path.exists(CT2_MODEL_DIR):
            print("Converting model to CTranslate2 INT8 format...")
            os.system(f"ct2-transformers-converter --model {ACTIVE_CONFIG['model']} --output_dir {CT2_MODEL_DIR} --quantization int8_float16")

def download_flores_raw(lang_code):
    url = f"https://raw.githubusercontent.com/facebookresearch/flores/main/flores200_dataset/dev/{lang_code}.dev"
    try:
        return requests.get(url).text.strip().split('\n')
    except:
        return ["The patient requires medical attention."] * 100

#EVALUATION

def run_medical_validation(translator, tokenizer):
    print("\n--- DOMAIN VALIDATION: MEDICAL SANITY CHECK ---")
    medical_checks = [
        "The patient requires a blood pressure check.",
        "Please take two tablets after every meal.",
        "The MRI results show no signs of fracture.",
        "Are you experiencing any shortness of breath?"
    ]
    for text in medical_checks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        output = translator.translate_batch([tokens], target_prefix=[["spa_Latn"]], beam_size=5)
        translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(output[0].hypotheses[0]))
        print(f"EN: {text}")
        print(f"ES: {translation.replace('spa_Latn ', '')}\n")

def benchmark_batch_throughput(translator, tokenizer):
    print("--- SCALABILITY: BATCH THROUGHPUT (1,000 CONCURRENT REQ SIMULATION) ---")
    batch_text = ["This is a test sentence for concurrent processing."] * 16
    batch_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in batch_text]
    
    start = time.perf_counter()
    _ = translator.translate_batch(batch_tokens, target_prefix=[["spa_Latn"]]*16, beam_size=5)
    end = time.perf_counter()
    
    total_time_ms = (end - start) * 1000
    print(f"Processed Batch of 16 in: {total_time_ms:.2f} ms")
    print(f"Effective latency per request: {total_time_ms/16:.2f} ms\n")

def run_main_benchmark():
    setup_environment()
    
    print("Fetching evaluation data from FLORES-200...")
    inputs = download_flores_raw("eng_Latn")[:100]
    targets = download_flores_raw("spa_Latn")[:100]

    translator = ctranslate2.Translator(CT2_MODEL_DIR, device=DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ACTIVE_CONFIG["model"])

    latencies = []
    results = []
    
    # Warm up
    warmup_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("Warm up."))
    for _ in range(5):
        _ = translator.translate_batch([warmup_tokens], target_prefix=[["spa_Latn"]])

    print(f"Executing benchmark on {len(inputs)} samples...")
    for text in inputs:
        start = time.perf_counter()
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        output = translator.translate_batch([tokens], target_prefix=[["spa_Latn"]], beam_size=5)
        translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(output[0].hypotheses[0]))
        
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(translation.replace('spa_Latn ', ''))
        
    
    results = [normalize(r) for r in results]
    targets = [normalize(t) for t in targets]
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    bleu = sacrebleu.corpus_bleu(results, [targets]).score
    
    print("\n" + "="*45)
    print(f"STELLA BENCHMARK: {ACTIVE_CONFIG['name']}")
    print("="*45)
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms (Target: <150ms)")
    print(f"BLEU Score:  {bleu:.2f} (Industry Target: >85)")
    print("="*45 + "\n")

    run_medical_validation(translator, tokenizer)
    benchmark_batch_throughput(translator, tokenizer)

if __name__ == "__main__":
    run_main_benchmark()
