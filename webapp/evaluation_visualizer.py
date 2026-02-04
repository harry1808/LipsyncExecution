"""
Visualization Module for Evaluation Results
Creates charts, tables, and HTML reports
"""

import json
from pathlib import Path
from typing import Dict, List
import base64
from io import BytesIO


def create_text_table(headers: List[str], rows: List[List], title: str = "") -> str:
    """
    Create ASCII table for console display.
    
    Args:
        headers: Column headers
        rows: List of rows
        title: Table title
    
    Returns:
        Formatted ASCII table string
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator
    separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    
    # Build table
    lines = []
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(separator))
    
    lines.append(separator)
    
    # Header row
    header_row = "|"
    for i, h in enumerate(headers):
        header_row += f" {h:<{col_widths[i]}} |"
    lines.append(header_row)
    lines.append(separator)
    
    # Data rows
    for row in rows:
        data_row = "|"
        for i, cell in enumerate(row):
            data_row += f" {str(cell):<{col_widths[i]}} |"
        lines.append(data_row)
    
    lines.append(separator)
    
    return "\n".join(lines)


def create_comparison_table(results: Dict) -> str:
    """
    Create side-by-side comparison table.
    
    Args:
        results: Evaluation results dictionary
    
    Returns:
        Formatted comparison table
    """
    tables = []
    
    # ASR Comparison
    if 'asr' in results.get('components', {}):
        asr = results['components']['asr']
        headers = ["Metric", "Value"]
        rows = [
            ["Word Error Rate (WER)", f"{asr['wer']:.2f}%"],
            ["Character Error Rate (CER)", f"{asr['cer']:.2f}%"],
            ["Accuracy", f"{asr['accuracy']:.2f}%"],
            ["Ground Truth", asr['ground_truth'][:50] + "..."],
            ["Recognized", asr['hypothesis'][:50] + "..."]
        ]
        tables.append(create_text_table(headers, rows, "SPEECH RECOGNITION RESULTS"))
    
    # Translation Comparison
    if 'translation' in results.get('components', {}):
        trans = results['components']['translation']
        headers = ["Metric", "Value"]
        rows = [
            ["BLEU Score", f"{trans['bleu_score']:.4f}"],
            ["BLEU-1", f"{trans['bleu_1']:.4f}"],
            ["BLEU-2", f"{trans['bleu_2']:.4f}"],
            ["BLEU-3", f"{trans['bleu_3']:.4f}"],
            ["BLEU-4", f"{trans['bleu_4']:.4f}"],
            ["Ground Truth", trans['ground_truth'][:50] + "..."],
            ["Translation", trans['hypothesis'][:50] + "..."]
        ]
        tables.append(create_text_table(headers, rows, "TRANSLATION RESULTS"))
    
    return "\n\n".join(tables)


def create_batch_summary_table(batch_results: Dict) -> str:
    """
    Create summary table for batch evaluation.
    
    Args:
        batch_results: Batch evaluation results
    
    Returns:
        Formatted summary table
    """
    lines = []
    lines.append("\n" + "="*70)
    lines.append("BATCH EVALUATION SUMMARY".center(70))
    lines.append("="*70)
    
    lines.append(f"\nTotal Test Cases: {batch_results['total_cases']}")
    lines.append(f"Successful: {batch_results['successful']}")
    lines.append(f"Failed: {batch_results['failed']}")
    lines.append(f"Success Rate: {batch_results['successful']/batch_results['total_cases']*100:.1f}%")
    
    # Aggregate metrics table
    if 'aggregate_metrics' in batch_results and batch_results['aggregate_metrics']:
        agg = batch_results['aggregate_metrics']
        
        headers = ["Metric", "Mean", "Std Dev", "Min", "Max"]
        rows = []
        
        if 'wer' in agg:
            rows.append([
                "WER (%)",
                f"{agg['wer']['mean']:.2f}",
                f"{agg['wer']['std']:.2f}",
                f"{agg['wer']['min']:.2f}",
                f"{agg['wer']['max']:.2f}"
            ])
        
        if 'bleu' in agg:
            rows.append([
                "BLEU",
                f"{agg['bleu']['mean']:.4f}",
                f"{agg['bleu']['std']:.4f}",
                f"{agg['bleu']['min']:.4f}",
                f"{agg['bleu']['max']:.4f}"
            ])
        
        if 'composite_score' in agg:
            rows.append([
                "Quality Score",
                f"{agg['composite_score']['mean']:.2f}",
                f"{agg['composite_score']['std']:.2f}",
                f"{agg['composite_score']['min']:.2f}",
                f"{agg['composite_score']['max']:.2f}"
            ])
        
        lines.append("\n" + create_text_table(headers, rows, "AGGREGATE METRICS"))
    
    # Individual results table
    headers = ["Test #", "Video", "Status", "WER", "BLEU", "Score"]
    rows = []
    
    for idx, result in enumerate(batch_results['individual_results'], 1):
        video_name = Path(result.get('video_path', 'N/A')).name
        status = "✓" if result.get('status') == 'success' else "✗"
        
        wer = "N/A"
        bleu = "N/A"
        score = "N/A"
        
        if result.get('status') == 'success':
            if 'asr' in result.get('components', {}):
                wer = f"{result['components']['asr']['wer']:.2f}%"
            if 'translation' in result.get('components', {}):
                bleu = f"{result['components']['translation']['bleu_score']:.4f}"
            if 'composite_score' in result:
                score = f"{result['composite_score']:.2f}"
        
        rows.append([
            f"{idx}",
            video_name[:20],
            status,
            wer,
            bleu,
            score
        ])
    
    lines.append("\n" + create_text_table(headers, rows, "INDIVIDUAL TEST RESULTS"))
    lines.append("\n" + "="*70 + "\n")
    
    return "\n".join(lines)


def create_html_report(results: Dict, output_path: str = "evaluation_report.html"):
    """
    Create detailed HTML report with styling.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save HTML file
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dubbing System Evaluation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .info-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .info-card p {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .section {{
            margin: 40px 0;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .metric-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .text-comparison {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        
        .text-label {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }}
        
        .text-content {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            font-family: monospace;
            line-height: 1.6;
        }}
        
        .score-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1.2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .score-excellent {{ background: #10b981; color: white; }}
        .score-good {{ background: #3b82f6; color: white; }}
        .score-fair {{ background: #f59e0b; color: white; }}
        .score-poor {{ background: #ef4444; color: white; }}
        
        .status-success {{
            color: #10b981;
            font-weight: bold;
        }}
        
        .status-error {{
            color: #ef4444;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Dubbing System Evaluation Report</h1>
            <p>Comprehensive Analysis of Video Dubbing Quality</p>
        </div>
        
        <div class="content">
"""

    # Add basic info
    status_class = "status-success" if results.get('status') == 'success' else "status-error"
    html_content += f"""
            <div class="info-grid">
                <div class="info-card">
                    <h3>Video File</h3>
                    <p>{Path(results.get('video_path', 'N/A')).name}</p>
                </div>
                <div class="info-card">
                    <h3>Languages</h3>
                    <p>{results.get('source_lang', 'N/A')} → {results.get('dest_lang', 'N/A')}</p>
                </div>
                <div class="info-card">
                    <h3>Status</h3>
                    <p class="{status_class}">{results.get('status', 'N/A').upper()}</p>
                </div>
                <div class="info-card">
                    <h3>Processing Time</h3>
                    <p>{results.get('total_processing_time', 0):.2f}s</p>
                </div>
            </div>
"""

    if results.get('status') == 'success':
        # Overall Score
        composite_score = results.get('composite_score', 0)
        score_class = (
            'score-excellent' if composite_score >= 90 else
            'score-good' if composite_score >= 75 else
            'score-fair' if composite_score >= 60 else
            'score-poor'
        )
        
        html_content += f"""
            <div class="section">
                <h2 class="section-title">Overall Quality Score</h2>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-name">Composite Score</span>
                        <span class="metric-value">{composite_score:.2f}/100</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {composite_score}%">
                            {composite_score:.1f}%
                        </div>
                    </div>
                    <span class="score-badge {score_class}">
                        {'⭐'*int(composite_score/20)} {_get_rating_text(composite_score)}
                    </span>
                </div>
            </div>
"""
        
        # ASR Results
        if 'asr' in results.get('components', {}):
            asr = results['components']['asr']
            accuracy = 100 - asr['wer']
            
            html_content += f"""
            <div class="section">
                <h2 class="section-title">1. Speech Recognition (ASR)</h2>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-name">Word Error Rate (WER)</span>
                        <span class="metric-value">{asr['wer']:.2f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {accuracy}%; background: {'#10b981' if asr['wer'] < 20 else '#f59e0b' if asr['wer'] < 40 else '#ef4444'}">
                            Accuracy: {accuracy:.1f}%
                        </div>
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Character Error Rate (CER)</td>
                        <td>{asr['cer']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Substitutions</td>
                        <td>{asr.get('substitutions', 0)}</td>
                    </tr>
                    <tr>
                        <td>Deletions</td>
                        <td>{asr.get('deletions', 0)}</td>
                    </tr>
                    <tr>
                        <td>Insertions</td>
                        <td>{asr.get('insertions', 0)}</td>
                    </tr>
                </table>
                
                <div class="text-comparison">
                    <div class="text-label">Ground Truth:</div>
                    <div class="text-content">{asr['ground_truth']}</div>
                </div>
                
                <div class="text-comparison">
                    <div class="text-label">Recognized:</div>
                    <div class="text-content">{asr['hypothesis']}</div>
                </div>
            </div>
"""
        
        # Translation Results
        if 'translation' in results.get('components', {}):
            trans = results['components']['translation']
            bleu_percent = trans['bleu_score'] * 100
            
            html_content += f"""
            <div class="section">
                <h2 class="section-title">2. Translation Quality</h2>
                
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-name">BLEU Score</span>
                        <span class="metric-value">{trans['bleu_score']:.4f}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {bleu_percent}%; background: {'#10b981' if trans['bleu_score'] > 0.4 else '#f59e0b' if trans['bleu_score'] > 0.2 else '#ef4444'}">
                            {bleu_percent:.1f}%
                        </div>
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>N-gram</th>
                        <th>Precision</th>
                    </tr>
                    <tr>
                        <td>BLEU-1 (Unigrams)</td>
                        <td>{trans['bleu_1']:.4f}</td>
                    </tr>
                    <tr>
                        <td>BLEU-2 (Bigrams)</td>
                        <td>{trans['bleu_2']:.4f}</td>
                    </tr>
                    <tr>
                        <td>BLEU-3 (Trigrams)</td>
                        <td>{trans['bleu_3']:.4f}</td>
                    </tr>
                    <tr>
                        <td>BLEU-4 (4-grams)</td>
                        <td>{trans['bleu_4']:.4f}</td>
                    </tr>
                </table>
                
                <div class="text-comparison">
                    <div class="text-label">Expected Translation:</div>
                    <div class="text-content">{trans['ground_truth']}</div>
                </div>
                
                <div class="text-comparison">
                    <div class="text-label">Generated Translation:</div>
                    <div class="text-content">{trans['hypothesis']}</div>
                </div>
            </div>
"""
        
        # Duration Results
        if 'duration' in results.get('components', {}):
            dur = results['components']['duration']
            
            html_content += f"""
            <div class="section">
                <h2 class="section-title">3. Duration Accuracy</h2>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Original Duration</td>
                        <td>{dur['ref_duration']:.2f}s</td>
                    </tr>
                    <tr>
                        <td>Dubbed Duration</td>
                        <td>{dur['syn_duration']:.2f}s</td>
                    </tr>
                    <tr>
                        <td>Difference</td>
                        <td>{dur['difference']:.2f}s</td>
                    </tr>
                    <tr>
                        <td>Error Percentage</td>
                        <td>{dur['error_percent']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Duration Ratio</td>
                        <td>{dur['ratio']:.3f}</td>
                    </tr>
                </table>
            </div>
"""
    
    else:
        # Error case
        html_content += f"""
            <div class="section">
                <div class="metric-card" style="border-left: 4px solid #ef4444;">
                    <h3 style="color: #ef4444;">Error</h3>
                    <p>{results.get('error', 'Unknown error occurred')}</p>
                </div>
            </div>
"""
    
    # Footer
    html_content += f"""
        </div>
        
        <div class="footer">
            <p>Report generated on {results.get('timestamp', 'N/A')}</p>
            <p>Video Dubbing System Evaluation Framework v1.0</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved to: {output_path}")


def _get_rating_text(score: float) -> str:
    """Get rating text from score."""
    if score >= 90:
        return "EXCELLENT"
    elif score >= 75:
        return "GOOD"
    elif score >= 60:
        return "FAIR"
    elif score >= 40:
        return "POOR"
    else:
        return "NEEDS IMPROVEMENT"


# Quick visualization function
def visualize_results(results_path: str, output_dir: str = "."):
    """
    Load results JSON and create visualizations.
    
    Args:
        results_path: Path to evaluation results JSON
        output_dir: Directory to save visualizations
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create HTML report
    html_path = output_dir / "evaluation_report.html"
    create_html_report(results, str(html_path))
    
    # Create text comparison
    comparison = create_comparison_table(results)
    print(comparison)
    
    return html_path


if __name__ == "__main__":
    print("Evaluation Visualizer")
    print("=" * 60)
    print("\nUsage:")
    print("""
    from webapp.evaluation_visualizer import visualize_results
    
    visualize_results(
        results_path="evaluation/evaluation_results.json",
        output_dir="./reports"
    )
    """)

