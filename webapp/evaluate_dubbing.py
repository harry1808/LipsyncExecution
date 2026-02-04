"""
Evaluation Pipeline for Video Dubbing System
Tests all components: ASR, Translation, TTS, End-to-End
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime

import numpy as np
from moviepy.editor import VideoFileClip

from .dubbing import recognize_speech, translate_text, process_video
from .evaluation_metrics import (
    calculate_bleu,
    calculate_wer,
    calculate_cer,
    calculate_duration_metrics,
    calculate_composite_score,
    format_metrics_report
)


class DubbingEvaluator:
    """
    Comprehensive evaluation system for video dubbing pipeline.
    
    Usage:
        evaluator = DubbingEvaluator()
        results = evaluator.evaluate_full_pipeline(
            video_path="test.mp4",
            ground_truth={
                'transcript': "original text",
                'translation': "translated text"
            }
        )
        print(results.get_report())
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.results = []
    
    def _setup_logger(self):
        """Setup default logger."""
        logger = logging.getLogger('DubbingEvaluator')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    # ========================================================================
    # COMPONENT-LEVEL EVALUATION
    # ========================================================================
    
    def evaluate_speech_recognition(
        self, 
        audio_path: str, 
        ground_truth: str,
        source_language: str
    ) -> Dict:
        """
        Evaluate speech recognition accuracy.
        
        Metrics:
            - WER (Word Error Rate)
            - CER (Character Error Rate)
            - Processing time
        
        Args:
            audio_path: Path to audio file
            ground_truth: Ground truth transcription
            source_language: Language code (e.g., 'en', 'hi')
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating Speech Recognition...")
        
        start_time = time.time()
        try:
            # Run ASR
            hypothesis = recognize_speech(audio_path, source_language, self.logger)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            wer_metrics = calculate_wer(ground_truth, hypothesis)
            cer_metrics = calculate_cer(ground_truth, hypothesis)
            
            result = {
                'component': 'speech_recognition',
                'ground_truth': ground_truth,
                'hypothesis': hypothesis,
                'wer': wer_metrics['wer'],
                'cer': cer_metrics['cer'],
                'accuracy': wer_metrics['accuracy'],
                'substitutions': wer_metrics['substitutions'],
                'deletions': wer_metrics['deletions'],
                'insertions': wer_metrics['insertions'],
                'processing_time': processing_time,
                'status': 'success'
            }
            
            self.logger.info(f"✓ ASR WER: {wer_metrics['wer']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"ASR evaluation failed: {e}")
            result = {
                'component': 'speech_recognition',
                'status': 'error',
                'error': str(e)
            }
        
        return result
    
    def evaluate_translation(
        self,
        source_text: str,
        ground_truth: str,
        source_lang: str,
        dest_lang: str
    ) -> Dict:
        """
        Evaluate translation quality.
        
        Metrics:
            - BLEU score (1-4 grams)
            - Processing time
        
        Args:
            source_text: Source language text
            ground_truth: Ground truth translation
            source_lang: Source language code
            dest_lang: Destination language code
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating Translation...")
        
        start_time = time.time()
        try:
            # Run translation
            hypothesis = translate_text(
                source_text, 
                source_lang, 
                dest_lang, 
                self.logger
            )
            processing_time = time.time() - start_time
            
            # Calculate BLEU
            bleu_metrics = calculate_bleu(hypothesis, ground_truth)
            
            result = {
                'component': 'translation',
                'source_text': source_text,
                'ground_truth': ground_truth,
                'hypothesis': hypothesis,
                'bleu_score': bleu_metrics['bleu_score'],
                'bleu_1': bleu_metrics['bleu_1'],
                'bleu_2': bleu_metrics['bleu_2'],
                'bleu_3': bleu_metrics['bleu_3'],
                'bleu_4': bleu_metrics['bleu_4'],
                'brevity_penalty': bleu_metrics['brevity_penalty'],
                'processing_time': processing_time,
                'status': 'success'
            }
            
            self.logger.info(f"✓ Translation BLEU: {bleu_metrics['bleu_score']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Translation evaluation failed: {e}")
            result = {
                'component': 'translation',
                'status': 'error',
                'error': str(e)
            }
        
        return result
    
    def evaluate_tts_duration(
        self,
        original_audio_path: str,
        synthesized_audio_path: str
    ) -> Dict:
        """
        Evaluate TTS duration accuracy.
        
        Metrics:
            - Duration difference
            - Duration ratio
            - Percentage error
        
        Args:
            original_audio_path: Original audio file
            synthesized_audio_path: Synthesized audio file
        
        Returns:
            Dictionary with duration metrics
        """
        self.logger.info("Evaluating TTS Duration...")
        
        try:
            from pydub import AudioSegment
            
            # Load audio files
            original = AudioSegment.from_file(original_audio_path)
            synthesized = AudioSegment.from_file(synthesized_audio_path)
            
            orig_duration = len(original) / 1000.0  # Convert to seconds
            synth_duration = len(synthesized) / 1000.0
            
            # Calculate metrics
            duration_metrics = calculate_duration_metrics(orig_duration, synth_duration)
            
            result = {
                'component': 'tts_duration',
                'original_duration': orig_duration,
                'synthesized_duration': synth_duration,
                'difference': duration_metrics['difference'],
                'ratio': duration_metrics['ratio'],
                'error_percent': duration_metrics['error_percent'],
                'status': 'success'
            }
            
            self.logger.info(f"✓ Duration Error: {duration_metrics['error_percent']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"TTS duration evaluation failed: {e}")
            result = {
                'component': 'tts_duration',
                'status': 'error',
                'error': str(e)
            }
        
        return result
    
    # ========================================================================
    # END-TO-END EVALUATION
    # ========================================================================
    
    def evaluate_full_pipeline(
        self,
        video_path: str,
        source_lang: str,
        dest_lang: str,
        ground_truth: Dict,
        output_dir: str = "./evaluation_output",
        voice: str = "female",
        enable_lipsync: bool = False
    ) -> Dict:
        """
        Evaluate complete dubbing pipeline end-to-end.
        
        Ground truth should contain:
            {
                'transcript': "original speech text",
                'translation': "expected translation"
            }
        
        Args:
            video_path: Path to input video
            source_lang: Source language code
            dest_lang: Destination language code
            ground_truth: Dictionary with ground truth data
            output_dir: Directory for output files
            voice: Voice gender ('male' or 'female')
            enable_lipsync: Whether to apply lip-sync
        
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("="*60)
        self.logger.info("STARTING END-TO-END DUBBING EVALUATION")
        self.logger.info("="*60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        overall_start = time.time()
        
        results = {
            'video_path': video_path,
            'source_lang': source_lang,
            'dest_lang': dest_lang,
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Get original video duration
            original_clip = VideoFileClip(video_path)
            original_duration = original_clip.duration
            original_clip.close()
            results['original_duration'] = original_duration
            
            # Run full dubbing pipeline
            self.logger.info("\n[1/3] Running dubbing pipeline...")
            dubbed_video_path, transcript, translation = process_video(
                video_path=video_path,
                source_lang=source_lang,
                dest_lang=dest_lang,
                output_dir=str(output_dir),
                logger=self.logger,
                voice=voice,
                enable_lipsync=enable_lipsync
            )
            
            results['dubbed_video_path'] = str(dubbed_video_path)
            results['generated_transcript'] = transcript
            results['generated_translation'] = translation
            
            # Evaluate Speech Recognition
            self.logger.info("\n[2/3] Evaluating Speech Recognition...")
            if 'transcript' in ground_truth:
                wer_metrics = calculate_wer(
                    ground_truth['transcript'], 
                    transcript
                )
                cer_metrics = calculate_cer(
                    ground_truth['transcript'],
                    transcript
                )
                
                results['components']['asr'] = {
                    'wer': wer_metrics['wer'],
                    'cer': cer_metrics['cer'],
                    'accuracy': wer_metrics['accuracy'],
                    'ground_truth': ground_truth['transcript'],
                    'hypothesis': transcript
                }
                self.logger.info(f"  WER: {wer_metrics['wer']:.2f}%")
            
            # Evaluate Translation
            self.logger.info("\n[3/3] Evaluating Translation...")
            if 'translation' in ground_truth:
                bleu_metrics = calculate_bleu(
                    translation,
                    ground_truth['translation']
                )
                
                results['components']['translation'] = {
                    'bleu_score': bleu_metrics['bleu_score'],
                    'bleu_1': bleu_metrics['bleu_1'],
                    'bleu_2': bleu_metrics['bleu_2'],
                    'bleu_3': bleu_metrics['bleu_3'],
                    'bleu_4': bleu_metrics['bleu_4'],
                    'ground_truth': ground_truth['translation'],
                    'hypothesis': translation
                }
                self.logger.info(f"  BLEU: {bleu_metrics['bleu_score']:.4f}")
            
            # Duration comparison
            dubbed_clip = VideoFileClip(str(dubbed_video_path))
            dubbed_duration = dubbed_clip.duration
            dubbed_clip.close()
            
            duration_metrics = calculate_duration_metrics(
                original_duration,
                dubbed_duration
            )
            
            results['components']['duration'] = duration_metrics
            
            # Calculate composite score
            composite_metrics = {
                'wer': results['components'].get('asr', {}).get('wer', 0),
                'bleu_score': results['components'].get('translation', {}).get('bleu_score', 0),
                'duration_error_percent': duration_metrics.get('error_percent', 0)
            }
            
            composite_score = calculate_composite_score(composite_metrics)
            results['composite_score'] = composite_score
            
            # Overall timing
            results['total_processing_time'] = time.time() - overall_start
            results['status'] = 'success'
            
            self.logger.info("\n" + "="*60)
            self.logger.info(f"COMPOSITE QUALITY SCORE: {composite_score:.2f}/100")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Pipeline evaluation failed: {e}", exc_info=True)
            results['status'] = 'error'
            results['error'] = str(e)
        
        # Save results
        self._save_results(results, output_dir)
        self.results.append(results)
        
        return results
    
    # ========================================================================
    # BATCH EVALUATION
    # ========================================================================
    
    def evaluate_batch(
        self,
        test_cases: List[Dict],
        output_dir: str = "./batch_evaluation"
    ) -> Dict:
        """
        Evaluate multiple test cases and aggregate results.
        
        Args:
            test_cases: List of test case dictionaries, each containing:
                {
                    'video_path': str,
                    'source_lang': str,
                    'dest_lang': str,
                    'ground_truth': {...}
                }
            output_dir: Output directory for results
        
        Returns:
            Aggregated results dictionary
        """
        self.logger.info(f"\nEvaluating {len(test_cases)} test cases...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_results = {
            'total_cases': len(test_cases),
            'successful': 0,
            'failed': 0,
            'individual_results': [],
            'aggregate_metrics': {}
        }
        
        for idx, test_case in enumerate(test_cases, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Test Case {idx}/{len(test_cases)}")
            self.logger.info(f"{'='*60}")
            
            result = self.evaluate_full_pipeline(
                video_path=test_case['video_path'],
                source_lang=test_case['source_lang'],
                dest_lang=test_case['dest_lang'],
                ground_truth=test_case['ground_truth'],
                output_dir=str(output_dir / f"test_{idx}"),
                voice=test_case.get('voice', 'female'),
                enable_lipsync=test_case.get('enable_lipsync', False)
            )
            
            batch_results['individual_results'].append(result)
            
            if result.get('status') == 'success':
                batch_results['successful'] += 1
            else:
                batch_results['failed'] += 1
        
        # Calculate aggregate metrics
        batch_results['aggregate_metrics'] = self._calculate_aggregate_metrics(
            batch_results['individual_results']
        )
        
        # Save batch results
        batch_output_file = output_dir / "batch_results.json"
        with open(batch_output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n✓ Batch results saved to: {batch_output_file}")
        
        return batch_results
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate mean and std of metrics across multiple results."""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {}
        
        # Collect metrics
        wer_scores = []
        bleu_scores = []
        composite_scores = []
        duration_errors = []
        
        for result in successful_results:
            if 'components' in result:
                if 'asr' in result['components']:
                    wer_scores.append(result['components']['asr']['wer'])
                if 'translation' in result['components']:
                    bleu_scores.append(result['components']['translation']['bleu_score'])
                if 'duration' in result['components']:
                    duration_errors.append(result['components']['duration']['error_percent'])
            if 'composite_score' in result:
                composite_scores.append(result['composite_score'])
        
        aggregate = {}
        
        if wer_scores:
            aggregate['wer'] = {
                'mean': np.mean(wer_scores),
                'std': np.std(wer_scores),
                'min': np.min(wer_scores),
                'max': np.max(wer_scores)
            }
        
        if bleu_scores:
            aggregate['bleu'] = {
                'mean': np.mean(bleu_scores),
                'std': np.std(bleu_scores),
                'min': np.min(bleu_scores),
                'max': np.max(bleu_scores)
            }
        
        if composite_scores:
            aggregate['composite_score'] = {
                'mean': np.mean(composite_scores),
                'std': np.std(composite_scores),
                'min': np.min(composite_scores),
                'max': np.max(composite_scores)
            }
        
        return aggregate
    
    def _save_results(self, results: Dict, output_dir: Path):
        """Save evaluation results to JSON file."""
        output_file = output_dir / "evaluation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n✓ Results saved to: {output_file}")
    
    def generate_report(self, results: Dict = None) -> str:
        """Generate human-readable evaluation report."""
        if results is None:
            if not self.results:
                return "No evaluation results available."
            results = self.results[-1]
        
        report_lines = []
        report_lines.append("\n" + "="*70)
        report_lines.append("VIDEO DUBBING SYSTEM - EVALUATION REPORT".center(70))
        report_lines.append("="*70)
        
        report_lines.append(f"\nVideo: {results.get('video_path', 'N/A')}")
        report_lines.append(f"Languages: {results.get('source_lang', 'N/A')} → {results.get('dest_lang', 'N/A')}")
        report_lines.append(f"Status: {results.get('status', 'N/A').upper()}")
        
        if results.get('status') == 'success':
            report_lines.append(f"\n{'─'*70}")
            report_lines.append("COMPONENT RESULTS:")
            report_lines.append(f"{'─'*70}")
            
            # ASR Results
            if 'asr' in results.get('components', {}):
                asr = results['components']['asr']
                report_lines.append("\n[1] Speech Recognition (ASR):")
                report_lines.append(f"    Word Error Rate (WER):  {asr['wer']:.2f}%")
                report_lines.append(f"    Character Error Rate:    {asr['cer']:.2f}%")
                report_lines.append(f"    Accuracy:                {asr['accuracy']:.2f}%")
                report_lines.append(f"\n    Ground Truth: {asr['ground_truth'][:80]}...")
                report_lines.append(f"    Recognized:   {asr['hypothesis'][:80]}...")
            
            # Translation Results
            if 'translation' in results.get('components', {}):
                trans = results['components']['translation']
                report_lines.append("\n[2] Translation:")
                report_lines.append(f"    BLEU Score:              {trans['bleu_score']:.4f}")
                report_lines.append(f"    BLEU-1 (unigrams):       {trans['bleu_1']:.4f}")
                report_lines.append(f"    BLEU-2 (bigrams):        {trans['bleu_2']:.4f}")
                report_lines.append(f"    BLEU-3 (trigrams):       {trans['bleu_3']:.4f}")
                report_lines.append(f"    BLEU-4 (4-grams):        {trans['bleu_4']:.4f}")
                report_lines.append(f"\n    Ground Truth: {trans['ground_truth'][:80]}...")
                report_lines.append(f"    Translation:  {trans['hypothesis'][:80]}...")
            
            # Duration Results
            if 'duration' in results.get('components', {}):
                dur = results['components']['duration']
                report_lines.append("\n[3] Duration Accuracy:")
                report_lines.append(f"    Original Duration:       {dur['ref_duration']:.2f}s")
                report_lines.append(f"    Dubbed Duration:         {dur['syn_duration']:.2f}s")
                report_lines.append(f"    Difference:              {dur['difference']:.2f}s")
                report_lines.append(f"    Error Percentage:        {dur['error_percent']:.2f}%")
            
            # Overall Score
            report_lines.append(f"\n{'─'*70}")
            report_lines.append("OVERALL QUALITY SCORE:")
            report_lines.append(f"{'─'*70}")
            composite = results.get('composite_score', 0)
            rating = self._get_quality_rating(composite)
            report_lines.append(f"\n    Score: {composite:.2f}/100")
            report_lines.append(f"    Rating: {rating}")
            
            # Processing Time
            if 'total_processing_time' in results:
                report_lines.append(f"\n    Total Processing Time: {results['total_processing_time']:.2f}s")
        
        else:
            report_lines.append(f"\nError: {results.get('error', 'Unknown error')}")
        
        report_lines.append("\n" + "="*70 + "\n")
        
        return '\n'.join(report_lines)
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert numeric score to quality rating."""
        if score >= 90:
            return "⭐⭐⭐⭐⭐ EXCELLENT"
        elif score >= 75:
            return "⭐⭐⭐⭐ GOOD"
        elif score >= 60:
            return "⭐⭐⭐ FAIR"
        elif score >= 40:
            return "⭐⭐ POOR"
        else:
            return "⭐ NEEDS IMPROVEMENT"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_evaluate(
    video_path: str,
    source_lang: str,
    dest_lang: str,
    transcript: str,
    translation: str,
    output_dir: str = "./evaluation"
) -> str:
    """
    Quick evaluation with minimal setup.
    
    Args:
        video_path: Path to video file
        source_lang: Source language code
        dest_lang: Destination language code
        transcript: Ground truth transcript
        translation: Ground truth translation
        output_dir: Output directory
    
    Returns:
        Evaluation report as string
    """
    evaluator = DubbingEvaluator()
    
    results = evaluator.evaluate_full_pipeline(
        video_path=video_path,
        source_lang=source_lang,
        dest_lang=dest_lang,
        ground_truth={
            'transcript': transcript,
            'translation': translation
        },
        output_dir=output_dir
    )
    
    return evaluator.generate_report(results)


if __name__ == "__main__":
    # Example usage
    print("Dubbing Evaluation System")
    print("=" * 60)
    print("\nUsage Example:")
    print("""
    from webapp.evaluate_dubbing import DubbingEvaluator
    
    evaluator = DubbingEvaluator()
    results = evaluator.evaluate_full_pipeline(
        video_path="test_video.mp4",
        source_lang="en",
        dest_lang="hi",
        ground_truth={
            'transcript': "Hello, how are you?",
            'translation': "नमस्ते, आप कैसे हैं?"
        }
    )
    
    print(evaluator.generate_report(results))
    """)

