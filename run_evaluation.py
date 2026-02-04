"""
Evaluation Runner Script
Load test cases from JSON and run comprehensive evaluation
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add webapp to path
sys.path.insert(0, str(Path(__file__).parent))

from webapp.evaluate_dubbing import DubbingEvaluator
from webapp.evaluation_visualizer import create_html_report, create_batch_summary_table


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def load_test_config(config_path: str) -> dict:
    """Load test configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_single_test(evaluator, test_case, eval_config):
    """Run evaluation on a single test case."""
    print(f"\n{'='*70}")
    print(f"Test: {test_case['name']}")
    print(f"ID: {test_case['id']}")
    print(f"{'='*70}")
    
    output_dir = Path(eval_config['output_dir']) / test_case['id']
    
    results = evaluator.evaluate_full_pipeline(
        video_path=test_case['video_path'],
        source_lang=test_case['source_lang'],
        dest_lang=test_case['dest_lang'],
        ground_truth=test_case['ground_truth'],
        output_dir=str(output_dir),
        voice=eval_config.get('voice_preference', 'female'),
        enable_lipsync=eval_config.get('enable_lipsync', False)
    )
    
    # Generate HTML report if requested
    if eval_config.get('generate_html_report', True):
        html_path = output_dir / f"{test_case['id']}_report.html"
        create_html_report(results, str(html_path))
        print(f"✓ HTML Report: {html_path}")
    
    return results


def run_batch_evaluation(config_path: str):
    """Run batch evaluation from config file."""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "VIDEO DUBBING SYSTEM - BATCH EVALUATION".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Load configuration
    config = load_test_config(config_path)
    test_cases = config['test_cases']
    eval_config = config['evaluation_config']
    
    print(f"Loaded {len(test_cases)} test cases from {config_path}")
    print(f"Output directory: {eval_config['output_dir']}\n")
    
    # Create evaluator
    evaluator = DubbingEvaluator()
    
    # Prepare test cases in format expected by evaluator
    formatted_test_cases = []
    for test in test_cases:
        formatted_test_cases.append({
            'video_path': test['video_path'],
            'source_lang': test['source_lang'],
            'dest_lang': test['dest_lang'],
            'ground_truth': test['ground_truth'],
            'voice': eval_config.get('voice_preference', 'female'),
            'enable_lipsync': eval_config.get('enable_lipsync', False)
        })
    
    # Run batch evaluation
    batch_results = evaluator.evaluate_batch(
        test_cases=formatted_test_cases,
        output_dir=eval_config['output_dir']
    )
    
    # Display summary
    print(create_batch_summary_table(batch_results))
    
    # Save detailed summary
    output_dir = Path(eval_config['output_dir'])
    summary_path = output_dir / "evaluation_summary.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(create_batch_summary_table(batch_results))
    
    print(f"\n✓ Evaluation Complete!")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"✓ Batch results: {output_dir / 'batch_results.json'}")
    
    return batch_results


def run_quick_evaluation(args):
    """Run quick evaluation on a single video."""
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "QUICK EVALUATION MODE".center(68) + "║")
    print("╚" + "═"*68 + "╝\n")
    
    evaluator = DubbingEvaluator()
    
    results = evaluator.evaluate_full_pipeline(
        video_path=args.video,
        source_lang=args.source_lang,
        dest_lang=args.dest_lang,
        ground_truth={
            'transcript': args.transcript,
            'translation': args.translation
        },
        output_dir=args.output or './quick_eval',
        voice=args.voice,
        enable_lipsync=args.lipsync
    )
    
    # Print report
    print(evaluator.generate_report(results))
    
    # Generate HTML
    if args.html:
        html_path = Path(args.output or './quick_eval') / "report.html"
        create_html_report(results, str(html_path))
        print(f"\n✓ HTML Report: {html_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate Video Dubbing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch evaluation from config file
  python run_evaluation.py --config test_data_template.json
  
  # Quick evaluation of single video
  python run_evaluation.py --quick \\
    --video test.mp4 \\
    --source-lang en \\
    --dest-lang hi \\
    --transcript "Hello world" \\
    --translation "नमस्ते दुनिया"
  
  # Run with HTML reports
  python run_evaluation.py --config test_data.json --html
  
  # Verbose logging
  python run_evaluation.py --config test_data.json --verbose
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to test configuration JSON file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick evaluation mode for single video'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Video path (quick mode only)'
    )
    
    parser.add_argument(
        '--source-lang',
        type=str,
        help='Source language code (quick mode only)'
    )
    
    parser.add_argument(
        '--dest-lang',
        type=str,
        help='Destination language code (quick mode only)'
    )
    
    parser.add_argument(
        '--transcript',
        type=str,
        help='Ground truth transcript (quick mode only)'
    )
    
    parser.add_argument(
        '--translation',
        type=str,
        help='Ground truth translation (quick mode only)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (quick mode only)'
    )
    
    parser.add_argument(
        '--voice',
        type=str,
        default='female',
        choices=['male', 'female'],
        help='Voice preference (default: female)'
    )
    
    parser.add_argument(
        '--lipsync',
        action='store_true',
        help='Enable lip-sync processing'
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML reports'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.quick:
            # Quick evaluation mode
            if not all([args.video, args.source_lang, args.dest_lang, 
                       args.transcript, args.translation]):
                parser.error("Quick mode requires: --video, --source-lang, "
                           "--dest-lang, --transcript, --translation")
            
            run_quick_evaluation(args)
            
        elif args.config:
            # Batch evaluation mode
            run_batch_evaluation(args.config)
            
        else:
            parser.print_help()
            print("\n⚠️  Please specify either --config or --quick mode\n")
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

