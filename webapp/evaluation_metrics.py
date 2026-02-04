"""
Evaluation Metrics for Dubbing System
Includes: BLEU, WER, CER, MCD, and quality scores
"""

import numpy as np
import re
from collections import Counter
from typing import List, Tuple, Dict
import logging


# ============================================================================
# 1. TRANSLATION QUALITY METRICS
# ============================================================================

def compute_ngrams(text: str, n: int) -> Counter:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text
        n: N-gram size (1=unigram, 2=bigram, etc.)
    
    Returns:
        Counter of n-grams
    
    Example:
        >>> compute_ngrams("the cat sat", 2)
        Counter({'the cat': 1, 'cat sat': 1})
    """
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def modified_precision(candidate: str, reference: str, n: int) -> float:
    """
    Calculate modified n-gram precision for BLEU.
    
    Formula:
        p_n = (Σ Count_clip(n-gram)) / (Σ Count(n-gram))
    
    Where Count_clip = min(Count(n-gram), Max_Ref_Count(n-gram))
    
    Args:
        candidate: Generated translation
        reference: Ground truth translation
        n: N-gram size
    
    Returns:
        Precision score between 0 and 1
    """
    candidate_ngrams = compute_ngrams(candidate, n)
    reference_ngrams = compute_ngrams(reference, n)
    
    if not candidate_ngrams:
        return 0.0
    
    # Clip counts to reference maximum
    clipped_counts = 0
    total_counts = 0
    
    for ngram, count in candidate_ngrams.items():
        clipped_counts += min(count, reference_ngrams.get(ngram, 0))
        total_counts += count
    
    if total_counts == 0:
        return 0.0
    
    return clipped_counts / total_counts


def brevity_penalty(candidate: str, reference: str) -> float:
    """
    Calculate brevity penalty for BLEU.
    
    Formula:
        BP = exp(1 - r/c)  if c <= r
        BP = 1             if c > r
    
    Where:
        r = reference length
        c = candidate length
    
    Args:
        candidate: Generated translation
        reference: Ground truth translation
    
    Returns:
        Penalty value (0 to 1)
    """
    c = len(candidate.split())
    r = len(reference.split())
    
    if c > r:
        return 1.0
    elif c == 0:
        return 0.0
    else:
        return np.exp(1 - r / c)


def calculate_bleu(candidate: str, reference: str, max_n: int = 4) -> Dict:
    """
    Calculate BLEU score for translation quality.
    
    Formula:
        BLEU = BP × exp(Σ w_n × log(p_n))
    
    Where:
        BP = Brevity Penalty
        w_n = uniform weights (1/N)
        p_n = modified n-gram precision
    
    Args:
        candidate: Generated translation
        reference: Ground truth translation  
        max_n: Maximum n-gram size (default: 4)
    
    Returns:
        Dictionary with BLEU scores and components
        
    Example:
        >>> ref = "the cat is on the mat"
        >>> cand = "the cat on the mat"
        >>> result = calculate_bleu(cand, ref)
        >>> print(f"BLEU-4: {result['bleu_score']:.3f}")
    """
    # Calculate precision for each n-gram
    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(candidate, reference, n)
        precisions.append(p)
    
    # Avoid log(0)
    precisions = [p if p > 0 else 1e-10 for p in precisions]
    
    # Calculate geometric mean with uniform weights
    weights = [1.0 / max_n] * max_n
    log_precisions = [w * np.log(p) for w, p in zip(weights, precisions)]
    geo_mean = np.exp(sum(log_precisions))
    
    # Apply brevity penalty
    bp = brevity_penalty(candidate, reference)
    bleu_score = bp * geo_mean
    
    return {
        'bleu_score': bleu_score,
        'bleu_1': precisions[0] if len(precisions) > 0 else 0,
        'bleu_2': precisions[1] if len(precisions) > 1 else 0,
        'bleu_3': precisions[2] if len(precisions) > 2 else 0,
        'bleu_4': precisions[3] if len(precisions) > 3 else 0,
        'brevity_penalty': bp,
        'length_ratio': len(candidate.split()) / max(len(reference.split()), 1)
    }


# ============================================================================
# 2. SPEECH RECOGNITION METRICS (WER, CER)
# ============================================================================

def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Calculate Levenshtein distance with operation counts.
    
    Uses Dynamic Programming:
        D[i,j] = min(
            D[i-1,j] + 1,      # deletion
            D[i,j-1] + 1,      # insertion
            D[i-1,j-1] + cost  # substitution
        )
    
    Args:
        ref: Reference tokens (words or characters)
        hyp: Hypothesis tokens
    
    Returns:
        Tuple of (distance, substitutions, deletions, insertions)
    """
    n, m = len(ref), len(hyp)
    
    # Initialize DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # Track operations
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    # Backtrack to count operations
    i, j = n, m
    subs, dels, ins = 0, 0, 0
    
    while i > 0 or j > 0:
        if i == 0:
            ins += j
            break
        elif j == 0:
            dels += i
            break
        elif ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        else:
            min_val = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            if dp[i-1][j-1] == min_val:
                subs += 1
                i -= 1
                j -= 1
            elif dp[i-1][j] == min_val:
                dels += 1
                i -= 1
            else:
                ins += 1
                j -= 1
    
    return dp[n][m], subs, dels, ins


def calculate_wer(reference: str, hypothesis: str) -> Dict:
    """
    Calculate Word Error Rate for speech recognition.
    
    Formula:
        WER = (S + D + I) / N × 100%
    
    Where:
        S = Substitutions
        D = Deletions
        I = Insertions
        N = Total words in reference
    
    Args:
        reference: Ground truth transcription
        hypothesis: ASR output
    
    Returns:
        Dictionary with WER and detailed metrics
        
    Example:
        >>> ref = "the cat sat on the mat"
        >>> hyp = "the cat on the mat"
        >>> result = calculate_wer(ref, hyp)
        >>> print(f"WER: {result['wer']:.2f}%")
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return {'wer': 0.0 if len(hyp_words) == 0 else 100.0}
    
    distance, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)
    
    wer = (distance / len(ref_words)) * 100
    
    return {
        'wer': wer,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'total_words': len(ref_words),
        'correct': len(ref_words) - (subs + dels),
        'accuracy': ((len(ref_words) - (subs + dels)) / len(ref_words)) * 100
    }


def calculate_cer(reference: str, hypothesis: str) -> Dict:
    """
    Calculate Character Error Rate.
    
    Formula:
        CER = (S + D + I) / N × 100%
    
    Same as WER but at character level.
    Useful for languages without clear word boundaries.
    
    Args:
        reference: Ground truth text
        hypothesis: Generated text
    
    Returns:
        Dictionary with CER and detailed metrics
    """
    ref_chars = list(reference.lower().replace(' ', ''))
    hyp_chars = list(hypothesis.lower().replace(' ', ''))
    
    if len(ref_chars) == 0:
        return {'cer': 0.0 if len(hyp_chars) == 0 else 100.0}
    
    distance, subs, dels, ins = levenshtein_distance(ref_chars, hyp_chars)
    
    cer = (distance / len(ref_chars)) * 100
    
    return {
        'cer': cer,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'total_chars': len(ref_chars),
        'correct': len(ref_chars) - (subs + dels),
        'accuracy': ((len(ref_chars) - (subs + dels)) / len(ref_chars)) * 100
    }


# ============================================================================
# 3. AUDIO QUALITY METRICS
# ============================================================================

def calculate_mcd_from_mfcc(ref_mfcc: np.ndarray, syn_mfcc: np.ndarray) -> float:
    """
    Calculate Mel-Cepstral Distortion between two MFCC arrays.
    
    Formula:
        MCD = (10/ln(10)) × sqrt(2 × Σ(c_k^ref - c_k^syn)²)
    
    Where:
        c_k = k-th mel-cepstral coefficient
        Factor 10/ln(10) converts to dB scale
    
    Args:
        ref_mfcc: Reference MFCC (shape: [frames, coeffs])
        syn_mfcc: Synthesized MFCC (shape: [frames, coeffs])
    
    Returns:
        MCD in dB (lower is better, <4 dB is good)
    """
    # Align frames (use shorter length)
    min_frames = min(ref_mfcc.shape[0], syn_mfcc.shape[0])
    ref_mfcc = ref_mfcc[:min_frames, 1:]  # Skip c0 (energy)
    syn_mfcc = syn_mfcc[:min_frames, 1:]
    
    # Calculate squared differences
    diff = ref_mfcc - syn_mfcc
    squared_diff = np.sum(diff ** 2, axis=1)
    
    # MCD formula
    mcd = (10 / np.log(10)) * np.sqrt(2 * np.mean(squared_diff))
    
    return mcd


def calculate_duration_metrics(ref_duration: float, syn_duration: float) -> Dict:
    """
    Calculate duration accuracy metrics.
    
    Args:
        ref_duration: Reference audio duration (seconds)
        syn_duration: Synthesized audio duration (seconds)
    
    Returns:
        Dictionary with duration metrics
    """
    diff = abs(ref_duration - syn_duration)
    ratio = syn_duration / ref_duration if ref_duration > 0 else 0
    error_percent = (diff / ref_duration * 100) if ref_duration > 0 else 0
    
    return {
        'ref_duration': ref_duration,
        'syn_duration': syn_duration,
        'difference': diff,
        'ratio': ratio,
        'error_percent': error_percent
    }


# ============================================================================
# 4. OVERALL QUALITY SCORES
# ============================================================================

def calculate_composite_score(metrics: Dict) -> float:
    """
    Calculate composite quality score from multiple metrics.
    
    Weighted average:
        Score = w1×(1-WER) + w2×BLEU + w3×(1-CER) + w4×AudioQuality
    
    Args:
        metrics: Dictionary containing evaluation metrics
    
    Returns:
        Composite score (0-100)
    """
    weights = {
        'transcription': 0.25,  # WER
        'translation': 0.30,     # BLEU
        'synthesis': 0.25,       # Audio quality
        'timing': 0.20           # Duration accuracy
    }
    
    score = 0.0
    
    # Transcription quality (inverse of WER)
    if 'wer' in metrics:
        transcription_score = max(0, 100 - metrics['wer'])
        score += weights['transcription'] * transcription_score
    
    # Translation quality (BLEU)
    if 'bleu_score' in metrics:
        translation_score = metrics['bleu_score'] * 100
        score += weights['translation'] * translation_score
    
    # Audio synthesis (inverse of CER)
    if 'cer' in metrics:
        synthesis_score = max(0, 100 - metrics['cer'])
        score += weights['synthesis'] * synthesis_score
    
    # Timing accuracy
    if 'duration_error_percent' in metrics:
        timing_score = max(0, 100 - metrics['duration_error_percent'])
        score += weights['timing'] * timing_score
    
    return score


def calculate_mos_statistics(ratings: List[float]) -> Dict:
    """
    Calculate Mean Opinion Score statistics.
    
    MOS Formula:
        MOS = (1/N) × Σ rating_i
    
    Args:
        ratings: List of ratings (1-5 scale)
    
    Returns:
        Dictionary with MOS and statistics
        
    Example:
        >>> ratings = [4, 5, 4, 3, 5, 4, 4]
        >>> mos = calculate_mos_statistics(ratings)
        >>> print(f"MOS: {mos['mos']:.2f} ± {mos['std']:.2f}")
    """
    if not ratings:
        return {'mos': 0.0}
    
    ratings = np.array(ratings)
    
    return {
        'mos': np.mean(ratings),
        'std': np.std(ratings),
        'median': np.median(ratings),
        'min': np.min(ratings),
        'max': np.max(ratings),
        'count': len(ratings),
        'distribution': {
            '5': np.sum(ratings == 5),
            '4': np.sum(ratings == 4),
            '3': np.sum(ratings == 3),
            '2': np.sum(ratings == 2),
            '1': np.sum(ratings == 1)
        }
    }


# ============================================================================
# 5. HELPER FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove punctuation (optional)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def format_metrics_report(metrics: Dict, title: str = "Evaluation Results") -> str:
    """
    Format metrics dictionary as readable report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        title: Report title
    
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append(f"{title:^60}")
    report.append("=" * 60)
    report.append("")
    
    # Translation metrics
    if 'bleu_score' in metrics:
        report.append("TRANSLATION QUALITY (BLEU):")
        report.append(f"  BLEU Score:        {metrics['bleu_score']:.4f}")
        report.append(f"  BLEU-1:            {metrics.get('bleu_1', 0):.4f}")
        report.append(f"  BLEU-2:            {metrics.get('bleu_2', 0):.4f}")
        report.append(f"  BLEU-3:            {metrics.get('bleu_3', 0):.4f}")
        report.append(f"  BLEU-4:            {metrics.get('bleu_4', 0):.4f}")
        report.append(f"  Brevity Penalty:   {metrics.get('brevity_penalty', 1):.4f}")
        report.append("")
    
    # WER metrics
    if 'wer' in metrics:
        report.append("SPEECH RECOGNITION ACCURACY (WER):")
        report.append(f"  WER:               {metrics['wer']:.2f}%")
        report.append(f"  Accuracy:          {metrics.get('accuracy', 0):.2f}%")
        report.append(f"  Substitutions:     {metrics.get('substitutions', 0)}")
        report.append(f"  Deletions:         {metrics.get('deletions', 0)}")
        report.append(f"  Insertions:        {metrics.get('insertions', 0)}")
        report.append("")
    
    # CER metrics
    if 'cer' in metrics:
        report.append("CHARACTER ERROR RATE:")
        report.append(f"  CER:               {metrics['cer']:.2f}%")
        report.append("")
    
    # Duration metrics
    if 'duration_error_percent' in metrics:
        report.append("TIMING ACCURACY:")
        report.append(f"  Duration Error:    {metrics['duration_error_percent']:.2f}%")
        report.append(f"  Duration Ratio:    {metrics.get('duration_ratio', 1):.3f}")
        report.append("")
    
    # Composite score
    if 'composite_score' in metrics:
        report.append("OVERALL QUALITY:")
        report.append(f"  Composite Score:   {metrics['composite_score']:.2f}/100")
        report.append("")
    
    report.append("=" * 60)
    
    return '\n'.join(report)

