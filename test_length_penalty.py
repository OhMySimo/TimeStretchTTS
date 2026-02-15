#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Length Penalty Test Script for Qwen3-TTS
Generates multiple clips with different penalty values for comparison.
"""

import torch
import soundfile as sf
import os
import time
import numpy as np
from pathlib import Path
from qwen_tts import Qwen3TTSModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test text (use the same for all tests for fair comparison)
TEST_TEXT = "Questa è la spada di Sirius Black, il mago malvagio fuggito da Azkaban."

# Reference audio for voice cloning
REF_AUDIO = "/content/audio_0001.wav"  # MODIFY THIS PATH

# Model
MODEL_PATH = "simone00/it2"

# Generation settings (your optimized values)
TEMPERATURE = 0.72
TOP_P = 0.875
MAX_NEW_TOKENS = 150

# Length penalty values to test
# 0.0 = disabled (baseline)
# 0.10 = gentle penalty
# 0.15 = medium penalty  
# 0.20 = strong penalty
# 0.25 = very strong penalty
PENALTY_VALUES = [0.0, 0.10, 0.15, 0.20, 0.25]

# Frames per text token values to test (optional, test one at a time)
# Lower = expects faster speech
# Higher = expects slower speech
FRAMES_PER_TOKEN_VALUES = [8.0]  # Start with default, adjust if needed

# Number of clips per configuration (to measure variance)
NUM_SAMPLES_PER_CONFIG = 3

# Output directory
OUTPUT_DIR = "length_penalty_test"

# ============================================================================
# SETUP
# ============================================================================

def measure_duration(wav, sr):
    """Measure audio duration in seconds"""
    return len(wav) / sr

def measure_speaking_rate(text, duration):
    """Estimate words per second"""
    words = len(text.split())
    return words / duration if duration > 0 else 0

def main():
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("🔬 Length Penalty Test Suite")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Text: {TEST_TEXT}")
    print(f"Reference Audio: {REF_AUDIO}")
    print(f"Temperature: {TEMPERATURE}, Top-P: {TOP_P}")
    print(f"Max Tokens: {MAX_NEW_TOKENS}")
    print("=" * 70)
    print()
    
    # Check if reference audio exists
    if not os.path.exists(REF_AUDIO):
        print(f"❌ ERROR: Reference audio not found: {REF_AUDIO}")
        print("Please update REF_AUDIO path in the script.")
        return
    
    # Load model once
    print("📥 Loading model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    print(f"✅ Model loaded on {device}")
    print()
    
    # Results storage
    results = []
    
    # Test each configuration
    total_tests = len(PENALTY_VALUES) * len(FRAMES_PER_TOKEN_VALUES) * NUM_SAMPLES_PER_CONFIG
    test_num = 0
    
    for frames_per_token in FRAMES_PER_TOKEN_VALUES:
        for alpha in PENALTY_VALUES:
            config_name = f"penalty_{alpha:.2f}_frames_{frames_per_token:.1f}"
            print(f"🧪 Testing: alpha={alpha:.2f}, frames_per_token={frames_per_token:.1f}")
            print("-" * 70)
            
            config_durations = []
            config_rates = []
            
            for sample_idx in range(NUM_SAMPLES_PER_CONFIG):
                test_num += 1
                print(f"  [{test_num}/{total_tests}] Generating sample {sample_idx + 1}...", end=" ")
                
                start_time = time.time()
                
                # Generate
                try:
                    wavs, sr = tts.generate_voice_clone(
                        text=TEST_TEXT,
                        language="Italian",
                        ref_audio=REF_AUDIO,
                        ref_text="",
                        max_new_tokens=MAX_NEW_TOKENS,
                        x_vector_only_mode=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        length_penalty_alpha=alpha,              # TEST PARAMETER
                        frames_per_text_token=frames_per_token,  # TEST PARAMETER
                    )
                    
                    gen_time = time.time() - start_time
                    
                    # Measure
                    wav = wavs[0]
                    duration = measure_duration(wav, sr)
                    speaking_rate = measure_speaking_rate(TEST_TEXT, duration)
                    
                    config_durations.append(duration)
                    config_rates.append(speaking_rate)
                    
                    # Save
                    filename = f"{config_name}_sample{sample_idx + 1}.wav"
                    filepath = output_path / filename
                    sf.write(filepath, wav, sr)
                    
                    print(f"✅ {duration:.2f}s ({speaking_rate:.2f} words/s) - saved")
                    
                except Exception as e:
                    print(f"❌ FAILED: {e}")
                    continue
            
            # Summarize config
            if config_durations:
                mean_duration = np.mean(config_durations)
                std_duration = np.std(config_durations)
                mean_rate = np.mean(config_rates)
                
                results.append({
                    'alpha': alpha,
                    'frames_per_token': frames_per_token,
                    'mean_duration': mean_duration,
                    'std_duration': std_duration,
                    'mean_rate': mean_rate,
                    'samples': NUM_SAMPLES_PER_CONFIG
                })
                
                print(f"  📊 Summary: {mean_duration:.2f}s ±{std_duration:.2f}s ({mean_rate:.2f} words/s)")
            print()
    
    # Final report
    print("=" * 70)
    print("📈 FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"{'Alpha':<8} {'Frames/Tok':<12} {'Duration':<15} {'Rate (w/s)':<12} {'Samples'}")
    print("-" * 70)
    
    for r in results:
        duration_str = f"{r['mean_duration']:.2f}s ±{r['std_duration']:.2f}"
        print(f"{r['alpha']:<8.2f} {r['frames_per_token']:<12.1f} {duration_str:<15} "
              f"{r['mean_rate']:<12.2f} {r['samples']}")
    
    print()
    print("=" * 70)
    print("✅ Test Complete!")
    print(f"📁 All clips saved in: {output_path.absolute()}")
    print()
    print("🎧 Next Steps:")
    print("1. Listen to the clips in order (penalty 0.0 → 0.25)")
    print("2. Compare duration and naturalness")
    print("3. Choose the alpha value with best duration/quality trade-off")
    print("4. If all clips are still too slow, try:")
    print("   - Lower frames_per_token (e.g., 7.0 or 6.5)")
    print("   - Higher alpha values (e.g., 0.30)")
    print()
    
    # Save results to CSV
    import csv
    csv_path = output_path / "results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['alpha', 'frames_per_token', 'mean_duration', 
                                                'std_duration', 'mean_rate', 'samples'])
        writer.writeheader()
        writer.writerows(results)
    print(f"📊 Results also saved to: {csv_path}")
    print()

if __name__ == "__main__":
    main()
