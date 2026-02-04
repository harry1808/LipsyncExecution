#!/usr/bin/env python3
"""
Diagnostic script to find why TTS stopped working
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("TTS DIAGNOSTIC TOOL")
print("=" * 60)

# 1. Check Python and package versions
print("\n[1] Python & Package Versions:")
print(f"  Python: {sys.version}")

try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError as e:
    print(f"  [ERROR] PyTorch not found: {e}")

try:
    import transformers
    print(f"  Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"  [ERROR] Transformers not found: {e}")

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    print(f"  [OK] Parler-TTS installed")
except ImportError as e:
    print(f"  [ERROR] Parler-TTS not found: {e}")

# 2. Check disk space
print("\n[2] Disk Space:")
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"  Total: {total // (2**30)} GB")
    print(f"  Used: {used // (2**30)} GB")
    print(f"  Free: {free // (2**30)} GB")
    if free < 5 * (2**30):  # Less than 5GB
        print(f"  [WARNING] Low disk space! TTS needs ~10GB for model cache")
except Exception as e:
    print(f"  [ERROR] Could not check disk space: {e}")

# 3. Check Hugging Face cache
print("\n[3] Hugging Face Model Cache:")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
print(f"  Cache location: {cache_dir}")
print(f"  Cache exists: {cache_dir.exists()}")

if cache_dir.exists():
    model_dirs = list(cache_dir.glob("models--*"))
    print(f"  Cached models: {len(model_dirs)}")
    
    # Check for Indic Parler TTS
    indic_tts = list(cache_dir.glob("models--ai4bharat--indic-parler-tts"))
    if indic_tts:
        model_path = indic_tts[0]
        print(f"  [OK] Indic Parler TTS found: {model_path}")
        
        # Check size
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        print(f"    Size: {total_size // (2**20)} MB")
        
        if total_size < 100 * (2**20):  # Less than 100MB
            print(f"    [WARNING] Cache seems incomplete (expected 3GB+)")
    else:
        print(f"  [WARNING] Indic Parler TTS not in cache (will download ~3GB)")

# 4. Check memory
print("\n[4] System Memory:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  Total: {mem.total // (2**30)} GB")
    print(f"  Available: {mem.available // (2**30)} GB")
    print(f"  Used: {mem.percent}%")
    if mem.available < 4 * (2**30):
        print(f"  [WARNING] Low memory! TTS needs 4-8GB free")
except ImportError:
    print(f"  (Install psutil to check: pip install psutil)")

# 5. Test model loading
print("\n[5] Testing Model Load:")
print("  Attempting to load tokenizer...")

try:
    from transformers import AutoTokenizer
    import time
    
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    elapsed = time.time() - start
    print(f"  [OK] Tokenizer loaded in {elapsed:.1f}s")
    
    print("\n  Attempting to load model (this may take a few minutes)...")
    print("  (Press Ctrl+C to skip if it takes too long)")
    
    from parler_tts import ParlerTTSForConditionalGeneration
    
    start = time.time()
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts")
    elapsed = time.time() - start
    
    print(f"  [OK] Model loaded in {elapsed:.1f}s")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) // 1_000_000}M")
    
except KeyboardInterrupt:
    print("\n  [SKIPPED] Interrupted by user")
except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# 6. Check internet connectivity
print("\n[6] Internet Connectivity:")
try:
    import urllib.request
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print("  [OK] Can reach huggingface.co")
except Exception as e:
    print(f"  [ERROR] Cannot reach huggingface.co: {e}")
    print(f"     (May cause issues if model not cached)")

# 7. Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

recommendations = []

if cache_dir.exists() and not indic_tts:
    recommendations.append("• Model not cached - first run will download 3GB")
    
if free < 10 * (2**30):
    recommendations.append("• Free up disk space (need at least 10GB)")
    
try:
    if mem.available < 4 * (2**30):
        recommendations.append("• Close other programs to free up RAM")
except:
    pass

if not torch.cuda.is_available():
    recommendations.append("• Using CPU - expect 5-10 min per video (GPU would be 30x faster)")

if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("• No obvious issues detected")
    print("• If still hanging, try: Clear cache and re-download model")

print("\n" + "=" * 60)
print("TO CLEAR CACHE AND START FRESH:")
print("=" * 60)
print(f"Delete: {cache_dir}")
print("Then restart Flask app to re-download")
print("\nOR run:")
print("  rm -rf ~/.cache/huggingface/hub/models--ai4bharat--indic-parler-tts")

print("\n[OK] Diagnostic complete!")

