# 📊 Ground Truth Best Practices for Evaluation

## ⚠️ Common Problem: Poor Scores with Google Translate

### Why Am I Getting Low Scores?

If you're using **Google Translate** as ground truth and seeing **poor BLEU scores** (e.g., 0.2-0.3), this is likely **NOT** because your system is bad, but because of **how BLEU works**.

---

## 🔬 Understanding the Problem

### What is BLEU Score?

**BLEU (Bilingual Evaluation Understudy)** is a metric that measures translation quality by comparing **exact n-gram matches** between your translation and a reference.

**Formula:**
```
BLEU = BP × exp(Σ w_n × log(p_n))

Where:
- p_n = precision of n-grams (1-word, 2-word, 3-word, 4-word sequences)
- BP = Brevity Penalty (penalizes translations that are too short)
- w_n = weights (typically 0.25 each)
```

### The Core Issue

**BLEU is VERY sensitive to:**
- Exact word choices
- Word order
- Phrasing style
- Synonym usage

Even if two translations are **semantically identical**, different wording results in **low BLEU scores**.

---

## 📉 Example: Why Scores Are Low

### Scenario

**Original English:** "Hello, how are you doing today?"

**Your System's Translation to Spanish:** "Hola, ¿cómo estás hoy?"

**Google Translate's Spanish:** "Hola, ¿cómo te va hoy?"

### BLEU Analysis

```
Your System:    "Hola, ¿cómo estás hoy?"
Ground Truth:   "Hola, ¿cómo te va hoy?"

1-gram matching:
  Matching: "Hola", "cómo", "hoy" = 3/4 words
  BLEU-1: 0.75

2-gram matching:
  System:  ["Hola cómo", "cómo estás", "estás hoy"]
  Reference: ["Hola cómo", "cómo te", "te va", "va hoy"]
  Matching: Only "Hola cómo" = 1/3 pairs
  BLEU-2: 0.33

3-gram matching:
  System: ["Hola cómo estás", "cómo estás hoy"]
  Reference: ["Hola cómo te", "cómo te va", "te va hoy"]
  Matching: 0/2
  BLEU-3: 0.00

Overall BLEU: ~0.28 (POOR!)
```

**But both translations are correct!** They just use different phrasing:
- "estás" (you are) vs "te va" (you're going/doing)
- Both are valid ways to ask "how are you" in Spanish

---

## ✅ Solution 1: Use System Output as Base (RECOMMENDED)

### The Best Approach

Instead of using Google Translate, use **your system's own translation** as the starting point:

1. **Run your dubbing system first**
2. **Copy the system's translation**
3. **Manually verify and correct only actual errors**
4. **Use this corrected version as ground truth**

### Why This Works

- Ensures consistent phrasing style
- Only penalizes actual errors, not stylistic differences
- Gives meaningful evaluation of system performance
- More closely matches real-world use cases

### Example Process

```
Step 1: System produces translation
  "Hola, ¿cómo estás hoy?"

Step 2: You verify it
  ✓ "Hola" = correct
  ✓ "cómo estás" = correct
  ✓ "hoy" = correct
  
Step 3: Use as ground truth (no changes needed)
  Ground Truth: "Hola, ¿cómo estás hoy?"

Result: BLEU = 1.0 (Perfect! Because there were no actual errors)
```

If there WAS an error:
```
Step 1: System produces translation
  "Hola, ¿cómo estas hoy?"  (missing accent on "estás")

Step 2: You correct it
  "Hola, ¿cómo estás hoy?"  (fixed accent)
  
Step 3: Use corrected version as ground truth
  Ground Truth: "Hola, ¿cómo estás hoy?"

Result: BLEU = ~0.95 (Slight penalty for the accent error - appropriate!)
```

---

## ✅ Solution 2: Multiple Reference Translations

If you want to use multiple reference translations (including Google Translate), you can modify the evaluation code:

### Standard BLEU (Single Reference)
```python
bleu = calculate_bleu(system_output, reference)
```

### Multi-Reference BLEU
```python
references = [
    "Hola, ¿cómo estás hoy?",      # Reference 1
    "Hola, ¿cómo te va hoy?",      # Reference 2 (Google Translate)
    "Hola, ¿qué tal estás hoy?"    # Reference 3
]

# Take maximum BLEU across all references
bleu = max(calculate_bleu(system_output, ref) for ref in references)
```

This is more forgiving but requires more work.

---

## ✅ Solution 3: Semantic Similarity Metrics

For a more semantic evaluation, consider adding **semantic similarity** metrics alongside BLEU:

### BERTScore
Measures semantic similarity using contextual embeddings:
```python
from bert_score import score

# More forgiving of paraphrases
P, R, F1 = score([system_translation], [reference], lang="es")
```

### COMET
Neural metric trained on human judgments:
```python
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("wmt20-comet-da"))
score = model.predict([{
    "src": source_text,
    "mt": system_translation,
    "ref": reference
}])
```

---

## 🎯 Practical Guidelines

### ✅ DO:

1. **Start with system output** and manually correct errors
2. **Use consistent terminology** across evaluations
3. **Verify with native speakers** when possible
4. **Keep similar phrasing** to the system unless it's wrong
5. **Document your ground truth creation process**

### ❌ DON'T:

1. **Use raw Google Translate** as ground truth
2. **Change correct translations** just because you prefer different words
3. **Mix formal/informal** styles arbitrarily
4. **Use synonyms** when the original word is correct
5. **Assume low BLEU** means bad translation

---

## 📊 Interpreting BLEU Scores

### With Proper Ground Truth (System-based):

| BLEU Score | Interpretation |
|------------|----------------|
| 0.9 - 1.0  | Excellent (minimal errors) |
| 0.7 - 0.9  | Good (some minor errors) |
| 0.5 - 0.7  | Fair (noticeable errors) |
| < 0.5      | Poor (significant errors) |

### With Google Translate as Ground Truth:

| BLEU Score | What It Actually Means |
|------------|------------------------|
| 0.7 - 1.0  | Exceptional (nearly identical phrasing) |
| 0.5 - 0.7  | Probably good, but different style |
| 0.3 - 0.5  | Could be good or bad (check manually) |
| < 0.3      | Likely different phrasing (not necessarily bad) |

**⚠️ Warning:** With Google Translate, scores of 0.3-0.5 might represent **perfectly correct** translations that just use different words!

---

## 🛠️ Using the Evaluation Helper

We've created an **Evaluation Helper** tool in the web interface:

1. Navigate to **Evaluate** section
2. Select your completed activity
3. Click **"Use Evaluation Helper (Recommended)"**
4. The helper shows:
   - Your system's translation
   - Side-by-side comparison
   - Easy copy buttons
   - Best practice guidelines

### Features:
- ✅ Copy system output with one click
- ✅ Edit inline to fix errors
- ✅ Clear guidance on what to verify
- ✅ Warnings about Google Translate issues

---

## 📝 Example Evaluation Workflow

### Bad Workflow (Low Scores):
```
1. Get Google Translate of source text
2. Use as ground truth
3. Run evaluation
4. Get BLEU = 0.25 😞
5. Think system is bad (it's not!)
```

### Good Workflow (Meaningful Scores):
```
1. Run your dubbing system
2. Use Evaluation Helper
3. Copy system's translation
4. Manually verify each sentence
5. Fix only actual errors
6. Run evaluation
7. Get BLEU = 0.88 😊
8. Know the 12% penalty reflects real errors
```

---

## 🔬 Technical Deep Dive: Why BLEU Fails for Style Differences

### N-gram Matching Example

**Reference:** "The quick brown fox jumps"
**Candidate 1:** "The fast brown fox jumps" (1 word different)
**Candidate 2:** "Quick brown fox jumps the" (all words present, wrong order)

```
Candidate 1:
  1-grams: 4/5 match (80%)
  2-grams: 2/4 match (50%)
  3-grams: 1/3 match (33%)
  BLEU ≈ 0.52

Candidate 2:
  1-grams: 5/5 match (100%)
  2-grams: 1/4 match (25%)
  3-grams: 0/3 match (0%)
  BLEU ≈ 0.00
```

**Candidate 1 is clearly better,** but BLEU doesn't capture this well because of n-gram rigidity.

### The Smoothing Problem

When n-grams don't match at all (like 3-grams in Candidate 2), BLEU can collapse to near-zero even if lower n-grams match well. This is why:
- **Single word changes** can cause large BLEU drops
- **Word reordering** is heavily penalized
- **Synonyms** are treated as errors

---

## 💡 Additional Metrics to Consider

### 1. Word Error Rate (WER)
Originally for speech recognition, but can be used for translation:
```
WER = (Substitutions + Deletions + Insertions) / Total Words
```

### 2. METEOR
More forgiving than BLEU, considers:
- Synonyms
- Stemming
- Paraphrases

### 3. TER (Translation Edit Rate)
Measures minimum edits needed to transform candidate into reference:
```
TER = (Edits Required) / (Reference Length)
```

### 4. chrF
Character-level matching, more robust to morphological differences:
```python
from sacrebleu import sentence_chrf
score = sentence_chrf(system_output, [reference])
```

---

## 🎓 Summary

### The Key Takeaway

**BLEU is a good metric ONLY when:**
- Ground truth matches the style/phrasing of your system
- You want to measure actual errors, not stylistic differences
- You understand its limitations

**For meaningful evaluation:**
1. ✅ Use your system's output as the base
2. ✅ Manually verify and correct errors
3. ✅ Use the Evaluation Helper tool
4. ❌ Don't use raw Google Translate
5. ❌ Don't assume low BLEU = bad translation

### Quick Reference

| Ground Truth Source | BLEU Interpretation | Recommended? |
|---------------------|---------------------|--------------|
| System output (corrected) | Accurate error measurement | ✅ YES |
| Google Translate | Artificial penalties for style | ❌ NO |
| Professional translator | Good if style matches | ✅ YES (expensive) |
| Multiple references | More forgiving | ✅ YES (time-consuming) |

---

## 🔗 Resources

- [BLEU Paper (Papineni et al., 2002)](https://aclanthology.org/P02-1040/)
- [Evaluation Helper Tool](/evaluate/helper) (in the web interface)
- [EVALUATION_README.md](./EVALUATION_README.md) - Full evaluation guide
- [sacrebleu documentation](https://github.com/mjpost/sacrebleu)

---

**Remember:** Evaluation metrics are tools to help you improve, not absolute judgments. Always manually verify results that seem surprising!

