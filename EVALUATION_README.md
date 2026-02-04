# 📊 Video Dubbing System - Evaluation Guide

## What You Can Evaluate and Show

This evaluation framework helps you measure and demonstrate the quality of your video dubbing system across multiple dimensions.

---

## 🎯 What to Evaluate

### 1. **Speech Recognition Accuracy (ASR)**
**What it measures:** How accurately the system transcribes the original audio

**Metrics you'll get:**
- **WER (Word Error Rate)**: Lower is better (0% = perfect)
  - `WER = (Substitutions + Deletions + Insertions) / Total Words × 100%`
  - **Good**: < 20%, **Excellent**: < 10%
  
- **CER (Character Error Rate)**: Character-level accuracy
  - Useful for languages without clear word boundaries
  
- **Accuracy**: Percentage of correctly recognized words

**Example Result:**
```
WER: 12.5%
Accuracy: 87.5%
Ground Truth: "Hello everyone welcome to this tutorial"
Recognized:   "Hello everyone welcome to the tutorial"
```

---

### 2. **Translation Quality**
**What it measures:** How accurately the system translates from source to target language

**Metrics you'll get:**
- **BLEU Score**: Translation quality (0-1 scale, higher is better)
  - `BLEU = BP × exp(Σ wₙ × log(pₙ))`
  - **Acceptable**: > 0.5, **Good**: > 0.7
  - Shows n-gram precision:
    - BLEU-1: Individual word accuracy
    - BLEU-2: Word pair accuracy
    - BLEU-3: Phrase accuracy (3 words)
    - BLEU-4: Phrase accuracy (4 words)

**Example Result:**
```
BLEU Score: 0.7234
BLEU-1: 0.85  (85% of words are correct)
BLEU-2: 0.76  (76% of word pairs match)
BLEU-3: 0.68
BLEU-4: 0.62
```

---

### 3. **Audio Duration Accuracy**
**What it measures:** How well the dubbed audio timing matches the original

**Metrics you'll get:**
- **Duration Error (%)**: Percentage difference in length
- **Duration Ratio**: Synthesized length / Original length
- **Absolute Difference**: Time difference in seconds

**Example Result:**
```
Original Duration: 15.5s
Dubbed Duration: 16.2s
Difference: 0.7s
Error: 4.5%
```

---

### 4. **Overall Quality Score**
**What it measures:** Composite quality score combining all metrics

**How it's calculated:**
```
Composite Score = 0.25×(100-WER) + 0.30×(BLEU×100) + 0.25×(100-CER) + 0.20×(100-DurError)
```

**Rating Scale:**
- 90-100: ⭐⭐⭐⭐⭐ **EXCELLENT** - Production ready
- 75-89:  ⭐⭐⭐⭐ **GOOD** - Minor improvements needed
- 60-74:  ⭐⭐⭐ **FAIR** - Acceptable for some use cases
- 40-59:  ⭐⭐ **POOR** - Needs significant work
- 0-39:   ⭐ **NEEDS IMPROVEMENT** - Major issues

---

## 🚀 How to Run Evaluation

### **Method 1: Quick Single Video Test**

```bash
python run_evaluation.py --quick \
  --video test_video.mp4 \
  --source-lang en \
  --dest-lang hi \
  --transcript "Hello everyone, welcome to this tutorial" \
  --translation "नमस्ते सभी को, इस ट्यूटोरियल में आपका स्वागत है" \
  --html
```

### **Method 2: Batch Evaluation (Multiple Videos)**

1. **Create test configuration file** (see `test_data_template.json`):
```json
{
  "test_cases": [
    {
      "id": "test_001",
      "video_path": "path/to/video1.mp4",
      "source_lang": "en",
      "dest_lang": "hi",
      "ground_truth": {
        "transcript": "Original text",
        "translation": "Expected translation"
      }
    }
  ]
}
```

2. **Run batch evaluation**:
```bash
python run_evaluation.py --config test_data_template.json --html
```

### **Method 3: Python API**

```python
from webapp.evaluate_dubbing import DubbingEvaluator

evaluator = DubbingEvaluator()

results = evaluator.evaluate_full_pipeline(
    video_path="test_video.mp4",
    source_lang="en",
    dest_lang="hi",
    ground_truth={
        'transcript': "Hello everyone",
        'translation': "नमस्ते सभी को"
    }
)

print(evaluator.generate_report(results))
```

---

## 📈 What Results You Get

### **1. Console Output (Real-time)**
```
==================================================================
VIDEO DUBBING SYSTEM - EVALUATION REPORT
==================================================================

Video: test_video.mp4
Languages: en → hi
Status: SUCCESS

──────────────────────────────────────────────────────────────────
COMPONENT RESULTS:
──────────────────────────────────────────────────────────────────

[1] Speech Recognition (ASR):
    Word Error Rate (WER):  12.50%
    Character Error Rate:    8.30%
    Accuracy:               87.50%

[2] Translation:
    BLEU Score:              0.7234
    BLEU-1 (unigrams):       0.8500
    BLEU-2 (bigrams):        0.7600

[3] Duration Accuracy:
    Original Duration:       15.50s
    Dubbed Duration:         16.20s
    Error Percentage:        4.50%

──────────────────────────────────────────────────────────────────
OVERALL QUALITY SCORE:
──────────────────────────────────────────────────────────────────

    Score: 82.30/100
    Rating: ⭐⭐⭐⭐ GOOD
```

### **2. JSON File (Machine Readable)**
Location: `evaluation_output/evaluation_results.json`

```json
{
  "video_path": "test_video.mp4",
  "source_lang": "en",
  "dest_lang": "hi",
  "status": "success",
  "components": {
    "asr": {
      "wer": 12.5,
      "cer": 8.3,
      "accuracy": 87.5
    },
    "translation": {
      "bleu_score": 0.7234,
      "bleu_1": 0.85
    }
  },
  "composite_score": 82.3
}
```

### **3. HTML Report (Beautiful Visual)**
Location: `evaluation_output/report.html`

Features:
- ✅ Colorful progress bars
- ✅ Side-by-side text comparison
- ✅ Metric breakdowns
- ✅ Score badges
- ✅ Professional formatting

![Sample HTML Report](docs/sample_report_preview.png)

### **4. Batch Summary (Multiple Videos)**
```
╔════════════════════════════════════════════════════════════════╗
║               BATCH EVALUATION SUMMARY                         ║
╠════════════════════════════════════════════════════════════════╣
║ Total Test Cases: 5                                            ║
║ Successful: 5                                                  ║
║ Success Rate: 100.0%                                          ║
╠════════════════════════════════════════════════════════════════╣
║ Metric        │ Mean   │ Std Dev │ Min    │ Max               ║
╠═══════════════╪════════╪═════════╪════════╪═══════════════════╣
║ WER (%)       │ 15.20  │ 2.60    │ 12.50  │ 18.90            ║
║ BLEU          │ 0.6880 │ 0.0300  │ 0.6520 │ 0.7230           ║
║ Quality Score │ 78.70  │ 3.30    │ 74.50  │ 82.30            ║
╚═══════════════╧════════╧═════════╧════════╧═══════════════════╝
```

---

## 📊 How to Present Results

### **For Academic Papers / Reports**

**Table Format:**
```
┌────────────────┬─────────┬─────────┬─────────────┬─────────────┐
│ Language Pair  │ WER (%) │  BLEU   │ Duration    │ Composite   │
│                │         │         │ Error (%)   │ Score       │
├────────────────┼─────────┼─────────┼─────────────┼─────────────┤
│ EN → HI        │  12.5   │  0.723  │     4.5     │    82.3     │
│ EN → TA        │  15.2   │  0.681  │     6.2     │    78.1     │
│ EN → BN        │  18.9   │  0.652  │     5.8     │    74.5     │
│ HI → EN        │  14.3   │  0.695  │     3.9     │    79.8     │
├────────────────┼─────────┼─────────┼─────────────┼─────────────┤
│ Average        │ 15.2±2.6│ 0.688±  │   5.1±1.0   │  78.7±3.3   │
│                │         │  0.030  │             │             │
└────────────────┴─────────┴─────────┴─────────────┴─────────────┘
```

**Text to Include in Paper:**
```
Our system achieved an average Word Error Rate of 15.2% (±2.6%) 
across 4 language pairs, indicating high accuracy in transcription. 
Translation quality measured by BLEU score averaged 0.688 (±0.030), 
demonstrating strong semantic preservation. The composite quality 
score of 78.7/100 indicates production-ready performance.
```

### **For Presentations**

**Key Highlights:**
- 🎯 **87.5% Transcription Accuracy** (WER: 12.5%)
- 🌐 **Quality Translation** (BLEU: 0.72)
- ⏱️ **Excellent Timing** (4.5% duration error)
- ⭐ **Overall Score: 82.3/100** (GOOD rating)

### **For Documentation**

Use the HTML report - it's:
- Professional looking
- Easy to understand
- Includes all details
- Shareable via browser

---

## 🧪 Best Practices

### **Test Data Preparation**
1. **Collect diverse videos**:
   - Different speakers (male/female)
   - Various audio quality levels
   - Multiple domains (education, news, casual)
   - Different lengths (5s to 60s)

2. **Create ground truth**:
   - Manually verify transcripts (critical!)
   - Use professional translators for translations
   - Or use verified MT with human review

3. **Minimum test set**:
   - 10-20 videos per language pair
   - Balance across different conditions

### **Running Evaluations**
1. Start with quick single tests
2. Debug any issues
3. Run full batch evaluation
4. Analyze aggregate metrics

### **Reporting Results**
1. Always show mean ± std deviation
2. Include min/max ranges
3. Compare with baselines if available
4. Highlight both strengths and limitations

---

## 📁 File Structure

```
lipsyncExecution/
├── webapp/
│   ├── evaluation_metrics.py      # Core metric calculations
│   ├── evaluate_dubbing.py        # Evaluation pipeline
│   └── evaluation_visualizer.py   # Report generation
├── example_evaluation.py          # Usage examples
├── run_evaluation.py              # CLI runner
├── test_data_template.json        # Test data template
├── EVALUATION_README.md           # This file
└── evaluation_output/             # Results (auto-created)
    ├── evaluation_results.json    # Detailed results
    ├── report.html                # Visual report
    └── batch_results.json         # Batch summary
```

---

## 🔍 Interpreting Metrics

### **WER (Word Error Rate)**
- **< 10%**: Excellent - Near-human accuracy
- **10-20%**: Good - Acceptable for most use cases
- **20-30%**: Fair - May need manual correction
- **> 30%**: Poor - System needs improvement

### **BLEU Score**
- **> 0.7**: Good - Captures meaning well
- **0.5-0.7**: Acceptable - Some semantic loss
- **0.3-0.5**: Fair - Significant differences
- **< 0.3**: Poor - Major translation issues

### **Duration Error**
- **< 5%**: Excellent - Natural timing
- **5-10%**: Good - Barely noticeable
- **10-20%**: Fair - May need video adjustment
- **> 20%**: Poor - Significant timing issues

---

## 🛠️ Troubleshooting

**Q: WER is very high (>30%)**
- Check audio quality
- Verify correct language code
- Test Whisper model directly
- Consider fine-tuning ASR model

**Q: BLEU score is low (<0.3)**
- Verify ground truth translation
- Check if domain-specific terminology
- Test NLLB model directly
- Consider using different translation model

**Q: Duration error is large (>20%)**
- Source and target languages may have different speech rates
- Check if TTS synthesis is too slow/fast
- May need to adjust TTS parameters

**Q: System crashes during evaluation**
- Check GPU memory availability
- Reduce batch size
- Verify all dependencies installed
- Check video file integrity

---

## 📚 References

**Metrics:**
- BLEU: [Papineni et al., 2002](https://aclanthology.org/P02-1040/)
- WER: Standard speech recognition metric
- MCD: [Kubichek, 1993]

**Models:**
- Whisper ASR: [OpenAI Whisper](https://github.com/openai/whisper)
- NLLB Translation: [Meta NLLB](https://github.com/facebookresearch/fairseq/tree/nllb)
- Indic TTS: [AI4Bharat](https://github.com/AI4Bharat/Indic-TTS)

---

## 💡 Quick Start

```bash
# 1. Run example to see what results look like
python example_evaluation.py

# 2. Create your test data file
cp test_data_template.json my_tests.json
# Edit my_tests.json with your video paths

# 3. Run evaluation
python run_evaluation.py --config my_tests.json --html

# 4. View results
# Open: evaluation_output/report.html in browser
```

---

## 📞 Support

For questions or issues:
1. Check `example_evaluation.py` for usage examples
2. Review this README
3. Check evaluation logs: `evaluation.log`

---

**Happy Evaluating! 🎉**

Show your results with confidence - the metrics are industry-standard and the reports are professional! ✨

