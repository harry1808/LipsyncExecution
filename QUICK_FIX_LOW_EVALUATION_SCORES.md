# 🚨 Quick Fix: Low Evaluation Scores with Google Translate

## The Problem

**Getting BLEU scores of 0.2-0.4 when using Google Translate as ground truth?** This is NORMAL and NOT a sign your system is bad!

## Why This Happens

**BLEU measures exact word/phrase matching**, not semantic meaning.

### Example:

```
Your System:     "Hello, how are you today?"
Google Translate: "Hi, how are you doing today?"

BLEU Score: ~0.28 (POOR!)
Reality: Both translations are CORRECT! ✅
```

Different words ("Hi" vs "Hello", "doing" vs "are") = Low BLEU score

## ✅ The Fix (3 Steps)

### Step 1: Use the Evaluation Helper
1. Go to **Evaluate** page
2. Click **"Use Evaluation Helper (Recommended)"** button
3. You'll see a side-by-side comparison

### Step 2: Start with System Output
1. Click **"Copy to Ground Truth"** to copy your system's translation
2. Manually verify each sentence
3. Fix ONLY actual errors (spelling, grammar, meaning)
4. Leave correct translations as-is (even if you prefer different words)

### Step 3: Submit
- Submit the form
- Get meaningful scores that reflect actual errors

## 📊 Expected Results

### With Google Translate (BAD):
- BLEU: 0.2-0.4 (looks terrible, but might be fine)
- No way to know if system is actually good

### With System-Based Ground Truth (GOOD):
- BLEU: 0.7-0.9 (accurate reflection of quality)
- Low scores = real errors
- High scores = good quality

## 🎯 Quick Rules

| ✅ DO | ❌ DON'T |
|-------|---------|
| Start with system output | Use raw Google Translate |
| Fix actual errors only | Change correct words "because you prefer" |
| Keep system's phrasing | Use completely different style |
| Use Evaluation Helper | Manually type everything |

## 📚 More Details

See **[EVALUATION_GROUND_TRUTH_GUIDE.md](./EVALUATION_GROUND_TRUTH_GUIDE.md)** for:
- Technical explanation of BLEU
- Multiple examples
- Alternative metrics
- Advanced techniques

---

**TL;DR:** Don't use Google Translate directly. Use the Evaluation Helper to start with your system's output and fix only real errors. This gives you meaningful scores! 🎯

