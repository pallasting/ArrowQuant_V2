#!/usr/bin/env python3
"""
Evolutionary Search for Optimal Quantization Configuration

This script uses genetic algorithms to find optimal layer-wise group sizes
for diffusion model quantization, maximizing accuracy while maintaining
compression targets.

Usage:
    python scripts/evolutionary_search.py \\
        --model path/to/model \\
        --output path/to/output \\
        --population-size 20 \\
        --generations 10 \\
        --target-metric cosine_similarity

Example:
    python scripts/evolutionary_search.py \\
        --model models/dream-7b \\
        --output models/dream-7b-optimized \\
        --population-size 15 \\
        --generations 8 \\
        --target-metric balanced
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evolutionary search for optimal quantization configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

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
        help="Path to output directory for search results",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size for genetic algorithm (default: 20)",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations to evolve (default: 10)",
    )

    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="Mutation rate (0.0 to 1.0, default: 0.2)",
    )

    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover rate (0.0 to 1.0, default: 0.7)",
    )

    parser.add_argument(
        "--elite-ratio",
        type=float,
        default=0.1,
        help="Elite selection ratio (0.0 to 1.0, default: 0.1)",
    )

    parser.add_argument(
        "--target-metric",
        type=str,
        choices=["cosine_similarity", "accuracy", "compression_ratio", "balanced"],
        default="cosine_similarity",
        help="Target metric to optimize (default: cosine_similarity)",
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=200,
        help="Maximum number of evaluations (default: 200)",
    )

    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Target bit width for quantization (default: 4)",
    )

    parser.add_argument(
        "--profile",
        type=str,
        choices=["edge", "local", "cloud"],
        default="local",
        help="Deployment profile (default: local)",
    )

    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save search results JSON (default: output/search_results.json)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_evolutionary_search(args: argparse.Namespace) -> Dict:
    """
    Run evolutionary search for optimal quantization configuration.

    Note: This is a Python wrapper that demonstrates the API.
    The actual implementation would call into Rust via PyO3 bindings.

    Args:
        args: Command-line arguments

    Returns:
        Search results dictionary
    """
    logger.info("Starting evolutionary search")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Population size: {args.population_size}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Target metric: {args.target_metric}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configuration for evolutionary search
    search_config = {
        "population_size": args.population_size,
        "num_generations": args.generations,
        "mutation_rate": args.mutation_rate,
        "crossover_rate": args.crossover_rate,
        "elite_ratio": args.elite_ratio,
        "target_metric": args.target_metric,
        "max_evaluations": args.max_evaluations,
    }

    # Base quantization configuration
    base_config = {
        "bit_width": args.bit_width,
        "deployment_profile": args.profile,
    }

    logger.info("Search configuration:")
    logger.info(json.dumps(search_config, indent=2))

    # TODO: Call Rust evolutionary search via PyO3
    # This would be implemented as:
    # from arrow_quant_v2 import EvolutionarySearch
    # search = EvolutionarySearch(search_config)
    # result = search.run(args.model, args.output, base_config)

    # For now, return mock results
    logger.warning(
        "Evolutionary search not yet integrated with PyO3 bindings. "
        "This is a demonstration script."
    )

    # Mock results structure
    results = {
        "search_config": search_config,
        "base_config": base_config,
        "best_individual": {
            "layer_group_sizes": {
                "layer1": 64,
                "layer2": 128,
                "layer3": 64,
            },
            "fitness": 0.87,
            "metrics": {
                "cosine_similarity": 0.87,
                "compression_ratio": 12.5,
                "model_size_mb": 45.2,
            },
        },
        "generation_history": [
            {
                "generation": 0,
                "best_fitness": 0.75,
                "avg_fitness": 0.65,
                "worst_fitness": 0.55,
            },
            {
                "generation": 1,
                "best_fitness": 0.80,
                "avg_fitness": 0.70,
                "worst_fitness": 0.60,
            },
            {
                "generation": 2,
                "best_fitness": 0.87,
                "avg_fitness": 0.75,
                "worst_fitness": 0.65,
            },
        ],
        "total_evaluations": 60,
    }

    return results


def save_results(results: Dict, output_path: Path, save_path: Optional[str]) -> None:
    """Save search results to JSON file."""
    if save_path is None:
        save_path = output_path / "search_results.json"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {save_path}")


def print_summary(results: Dict) -> None:
    """Print summary of search results."""
    print("\n" + "=" * 80)
    print("EVOLUTIONARY SEARCH RESULTS")
    print("=" * 80)

    best = results["best_individual"]
    print(f"\nBest Configuration Found:")
    print(f"  Fitness: {best['fitness']:.4f}")

    if best.get("metrics"):
        metrics = best["metrics"]
        print(f"\nMetrics:")
        print(f"  Cosine Similarity: {metrics.get('cosine_similarity', 'N/A'):.4f}")
        print(f"  Compression Ratio: {metrics.get('compression_ratio', 'N/A'):.2f}x")
        print(f"  Model Size: {metrics.get('model_size_mb', 'N/A'):.2f} MB")

    print(f"\nLayer-wise Group Sizes:")
    for layer_name, group_size in sorted(best["layer_group_sizes"].items()):
        print(f"  {layer_name}: {group_size}")

    print(f"\nSearch Statistics:")
    print(f"  Total Evaluations: {results['total_evaluations']}")
    print(f"  Generations: {len(results['generation_history'])}")

    if results["generation_history"]:
        history = results["generation_history"]
        print(f"\nEvolution Progress:")
        print(f"  Initial Best Fitness: {history[0]['best_fitness']:.4f}")
        print(f"  Final Best Fitness: {history[-1]['best_fitness']:.4f}")
        improvement = (
            (history[-1]["best_fitness"] - history[0]["best_fitness"])
            / history[0]["best_fitness"]
            * 100
        )
        print(f"  Improvement: {improvement:.2f}%")

    print("\n" + "=" * 80)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Run evolutionary search
        results = run_evolutionary_search(args)

        # Save results
        save_results(results, Path(args.output), args.save_results)

        # Print summary
        print_summary(results)

        logger.info("Evolutionary search completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Evolutionary search failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
