#!/usr/bin/env python3
"""
Establish Baseline Thermodynamic Metrics

This script quantizes a model and establishes baseline Markov smoothness metrics
for the thermodynamic enhancement feature. It documents:
- Baseline smoothness score (expected ~0.65-0.78 for Dream 7B)
- Common violation patterns
- Per-boundary smoothness scores

Usage:
    # Run on Dream 7B (or any available model)
    python scripts/establish_baseline_metrics.py \\
        --model dream-7b/ \\
        --output dream-7b-int2-baseline/ \\
        --bit-width 2 \\
        --report baseline_metrics.json

    # Run with custom time groups
    python scripts/establish_baseline_metrics.py \\
        --model dream-7b/ \\
        --output dream-7b-int2-baseline/ \\
        --bit-width 2 \\
        --num-time-groups 4 \\
        --report baseline_metrics.json

Author: ArrowQuant V2 Team
License: MIT
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("baseline_metrics.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Baseline thermodynamic metrics."""
    
    model_name: str
    bit_width: int
    num_time_groups: int
    
    # Markov smoothness metrics
    smoothness_score: float
    violation_count: int
    violations: List[Dict[str, Any]]
    boundary_scores: List[float]
    
    # Quantization quality metrics
    cosine_similarity: float
    compression_ratio: float
    model_size_mb: float
    
    # Timing
    quantization_time_s: float
    
    # Statistics
    min_boundary_score: float
    max_boundary_score: float
    mean_boundary_score: float
    std_boundary_score: float
    
    # Common patterns
    common_violation_patterns: List[str]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Establish Baseline Thermodynamic Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on Dream 7B
  %(prog)s --model dream-7b/ --output dream-7b-int2-baseline/ --bit-width 2

  # Run with custom configuration
  %(prog)s --model dream-7b/ --output dream-7b-int2-baseline/ \\
           --bit-width 2 --num-time-groups 4 --report baseline.json

  # Run on test model
  %(prog)s --model tests/fixtures/test-model/ --output test-baseline/ \\
           --bit-width 2 --report test_baseline.json

For more information, see: .kiro/specs/thermodynamic-enhancement/README.md
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to input model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output quantized model directory",
    )

    # Quantization parameters
    parser.add_argument(
        "--bit-width",
        type=int,
        default=2,
        choices=[2, 4, 8],
        help="Quantization bit width (default: 2)",
    )
    parser.add_argument(
        "--num-time-groups",
        type=int,
        default=4,
        help="Number of time groups for time-aware quantization (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="Group size for spatial quantization (optional)",
    )

    # Thermodynamic validation parameters
    parser.add_argument(
        "--smoothness-threshold",
        type=float,
        default=0.3,
        help="Smoothness threshold for violation detection (default: 0.3 = 30%%)",
    )

    # Output parameters
    parser.add_argument(
        "--report",
        type=str,
        default="baseline_metrics.json",
        help="Path to save baseline metrics report (default: baseline_metrics.json)",
    )
    parser.add_argument(
        "--html-report",
        type=str,
        help="Path to save HTML report (optional)",
    )

    # Behavior parameters
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def analyze_violation_patterns(violations: List[Dict[str, Any]]) -> List[str]:
    """Analyze violations to identify common patterns."""
    patterns = []
    
    if not violations:
        patterns.append("No violations detected - excellent smoothness")
        return patterns
    
    # Analyze severity distribution
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    for v in violations:
        severity_counts[v["severity"]] += 1
    
    if severity_counts["high"] > 0:
        patterns.append(
            f"High severity violations: {severity_counts['high']} "
            f"({severity_counts['high']/len(violations)*100:.1f}%)"
        )
    if severity_counts["medium"] > 0:
        patterns.append(
            f"Medium severity violations: {severity_counts['medium']} "
            f"({severity_counts['medium']/len(violations)*100:.1f}%)"
        )
    if severity_counts["low"] > 0:
        patterns.append(
            f"Low severity violations: {severity_counts['low']} "
            f"({severity_counts['low']/len(violations)*100:.1f}%)"
        )
    
    # Analyze boundary positions
    boundary_indices = [v["boundary_idx"] for v in violations]
    if boundary_indices:
        patterns.append(
            f"Violations at boundaries: {sorted(set(boundary_indices))}"
        )
        
        # Check if violations cluster at early/late boundaries
        if all(idx < len(violations) / 2 for idx in boundary_indices):
            patterns.append("Violations cluster at early boundaries")
        elif all(idx >= len(violations) / 2 for idx in boundary_indices):
            patterns.append("Violations cluster at late boundaries")
    
    # Analyze jump magnitudes
    scale_jumps = [v["scale_jump"] for v in violations]
    if scale_jumps:
        max_jump = max(scale_jumps)
        avg_jump = sum(scale_jumps) / len(scale_jumps)
        patterns.append(
            f"Scale jumps: max={max_jump*100:.1f}%, avg={avg_jump*100:.1f}%"
        )
    
    return patterns


def establish_baseline(args: argparse.Namespace) -> BaselineMetrics:
    """Establish baseline metrics by quantizing model with validation enabled."""
    try:
        # Import ArrowQuant V2
        try:
            from arrow_quant_v2 import ArrowQuantV2
        except ImportError as e:
            logger.error(
                "Failed to import arrow_quant_v2. "
                "Please install with: pip install -e ."
            )
            raise

        # Validate paths
        model_path = Path(args.model)
        output_path = Path(args.output)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")

        logger.info("=" * 80)
        logger.info("ESTABLISHING BASELINE THERMODYNAMIC METRICS")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Bit width: INT{args.bit_width}")
        logger.info(f"Time groups: {args.num_time_groups}")
        logger.info(f"Smoothness threshold: {args.smoothness_threshold}")
        logger.info("=" * 80)

        # Create quantizer with thermodynamic validation enabled
        logger.info("\nInitializing quantizer with thermodynamic validation...")
        quantizer = ArrowQuantV2(mode="diffusion")

        # Quantize model with validation
        logger.info("\nQuantizing model (this may take several minutes)...")
        start_time = time.time()
        
        result = quantizer.quantize_diffusion_model(
            model_path=str(model_path),
            output_path=str(output_path),
            bit_width=args.bit_width,
            num_time_groups=args.num_time_groups,
            group_size=args.group_size,
            enable_thermodynamic_validation=True,
            smoothness_threshold=args.smoothness_threshold,
        )
        
        quantization_time = time.time() - start_time

        # Extract thermodynamic metrics
        logger.info("\nExtracting thermodynamic metrics...")
        thermo_metrics = result.get("thermodynamic_metrics", {})
        
        smoothness_score = thermo_metrics.get("smoothness_score", 0.0)
        violations = thermo_metrics.get("violations", [])
        boundary_scores = thermo_metrics.get("boundary_scores", [])
        
        # Compute statistics
        if boundary_scores:
            min_score = min(boundary_scores)
            max_score = max(boundary_scores)
            mean_score = sum(boundary_scores) / len(boundary_scores)
            std_score = (
                sum((x - mean_score) ** 2 for x in boundary_scores) / len(boundary_scores)
            ) ** 0.5
        else:
            min_score = max_score = mean_score = std_score = 0.0

        # Analyze violation patterns
        common_patterns = analyze_violation_patterns(violations)

        # Create baseline metrics
        baseline = BaselineMetrics(
            model_name=model_path.name,
            bit_width=args.bit_width,
            num_time_groups=args.num_time_groups,
            smoothness_score=smoothness_score,
            violation_count=len(violations),
            violations=violations,
            boundary_scores=boundary_scores,
            cosine_similarity=result.get("cosine_similarity", 0.0),
            compression_ratio=result.get("compression_ratio", 0.0),
            model_size_mb=result.get("model_size_mb", 0.0),
            quantization_time_s=quantization_time,
            min_boundary_score=min_score,
            max_boundary_score=max_score,
            mean_boundary_score=mean_score,
            std_boundary_score=std_score,
            common_violation_patterns=common_patterns,
        )

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE METRICS ESTABLISHED")
        logger.info("=" * 80)
        logger.info(f"\nMarkov Smoothness Metrics:")
        logger.info(f"  Overall smoothness score: {smoothness_score:.4f}")
        logger.info(f"  Violation count: {len(violations)}")
        logger.info(f"  Smoothness threshold: {args.smoothness_threshold}")
        
        logger.info(f"\nBoundary Score Statistics:")
        logger.info(f"  Min: {min_score:.4f}")
        logger.info(f"  Max: {max_score:.4f}")
        logger.info(f"  Mean: {mean_score:.4f}")
        logger.info(f"  Std: {std_score:.4f}")
        logger.info(f"  Total boundaries: {len(boundary_scores)}")
        
        logger.info(f"\nQuantization Quality:")
        logger.info(f"  Cosine similarity: {baseline.cosine_similarity:.4f}")
        logger.info(f"  Compression ratio: {baseline.compression_ratio:.2f}x")
        logger.info(f"  Model size: {baseline.model_size_mb:.2f} MB")
        logger.info(f"  Quantization time: {quantization_time:.2f}s")
        
        logger.info(f"\nCommon Violation Patterns:")
        for pattern in common_patterns:
            logger.info(f"  - {pattern}")
        
        if violations:
            logger.info(f"\nDetailed Violations:")
            for i, v in enumerate(violations[:5], 1):  # Show first 5
                logger.info(
                    f"  {i}. Boundary {v['boundary_idx']}: "
                    f"{v['scale_jump']*100:.1f}% scale jump, "
                    f"{v['zero_point_jump']*100:.1f}% zero_point jump "
                    f"({v['severity']} severity)"
                )
            if len(violations) > 5:
                logger.info(f"  ... and {len(violations) - 5} more violations")
        
        logger.info("=" * 80)

        return baseline

    except Exception as e:
        logger.error(f"Failed to establish baseline: {e}")
        logger.exception("Full traceback:")
        raise


def save_json_report(baseline: BaselineMetrics, output_path: str):
    """Save baseline metrics to JSON file."""
    try:
        report = asdict(baseline)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nBaseline metrics saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON report: {e}")


def save_html_report(baseline: BaselineMetrics, output_path: str):
    """Save baseline metrics to HTML file."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Baseline Thermodynamic Metrics - {baseline.model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .score-indicator {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .score-good {{
            background-color: #d4edda;
            color: #155724;
        }}
        .score-medium {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .score-poor {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pattern-list {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 10px 0;
        }}
        .pattern-list li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Baseline Thermodynamic Metrics</h1>
        <p><strong>Model:</strong> {baseline.model_name} | <strong>Bit Width:</strong> INT{baseline.bit_width} | <strong>Time Groups:</strong> {baseline.num_time_groups}</p>
        
        <div class="score-indicator {'score-good' if baseline.smoothness_score >= 0.78 else 'score-medium' if baseline.smoothness_score >= 0.65 else 'score-poor'}">
            Smoothness Score: {baseline.smoothness_score:.4f}
        </div>
        
        <h2>Markov Smoothness Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Smoothness Score</div>
                <div class="metric-value">{baseline.smoothness_score:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Violation Count</div>
                <div class="metric-value">{baseline.violation_count}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Boundaries</div>
                <div class="metric-value">{len(baseline.boundary_scores)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Violation Rate</div>
                <div class="metric-value">{baseline.violation_count/max(len(baseline.boundary_scores), 1)*100:.1f}%</div>
            </div>
        </div>
        
        <h2>Boundary Score Statistics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Minimum</div>
                <div class="metric-value">{baseline.min_boundary_score:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Maximum</div>
                <div class="metric-value">{baseline.max_boundary_score:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Mean</div>
                <div class="metric-value">{baseline.mean_boundary_score:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Std Dev</div>
                <div class="metric-value">{baseline.std_boundary_score:.4f}</div>
            </div>
        </div>
        
        <h2>Quantization Quality</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Cosine Similarity</div>
                <div class="metric-value">{baseline.cosine_similarity:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Compression Ratio</div>
                <div class="metric-value">{baseline.compression_ratio:.2f}x</div>
            </div>
            <div class="metric">
                <div class="metric-label">Model Size</div>
                <div class="metric-value">{baseline.model_size_mb:.1f} MB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Quantization Time</div>
                <div class="metric-value">{baseline.quantization_time_s:.1f}s</div>
            </div>
        </div>
        
        <h2>Common Violation Patterns</h2>
        <div class="pattern-list">
            <ul>
"""
    
    for pattern in baseline.common_violation_patterns:
        html += f"                <li>{pattern}</li>\n"
    
    html += """            </ul>
        </div>
"""
    
    if baseline.violations:
        html += """
        <h2>Detailed Violations</h2>
        <table>
            <tr>
                <th>Boundary Index</th>
                <th>Scale Jump</th>
                <th>Zero Point Jump</th>
                <th>Severity</th>
            </tr>
"""
        for v in baseline.violations:
            html += f"""            <tr>
                <td>{v['boundary_idx']}</td>
                <td>{v['scale_jump']*100:.2f}%</td>
                <td>{v['zero_point_jump']*100:.2f}%</td>
                <td>{v['severity']}</td>
            </tr>
"""
        html += """        </table>
"""
    
    html += """    </div>
</body>
</html>
"""
    
    try:
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save HTML report: {e}")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Establish baseline
        baseline = establish_baseline(args)

        # Save reports
        save_json_report(baseline, args.report)
        
        if args.html_report:
            save_html_report(baseline, args.html_report)

        # Print interpretation
        logger.info("\n" + "=" * 80)
        logger.info("INTERPRETATION")
        logger.info("=" * 80)
        
        if baseline.smoothness_score >= 0.78:
            logger.info("✓ Excellent smoothness - above expected baseline (0.78)")
        elif baseline.smoothness_score >= 0.65:
            logger.info("✓ Good smoothness - within expected baseline range (0.65-0.78)")
        else:
            logger.info("⚠ Below expected baseline - consider Phase 2 smoothing")
        
        logger.info(f"\nExpected improvements with thermodynamic enhancement:")
        logger.info(f"  Phase 2 (Boundary Smoothing): ~0.82+ smoothness, +2-3% accuracy")
        logger.info(f"  Phase 3 (Optimization): ~0.90+ smoothness, +6-8% accuracy")
        logger.info("=" * 80)

        logger.info("\nBaseline establishment completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Failed to establish baseline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
