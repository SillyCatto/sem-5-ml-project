"""
CLI entry point for running the preprocessing pipeline.

Usage:
    uv run python -m preprocessing                  # process full dataset
    uv run python -m preprocessing --skip-flow       # skip optical flow (fast)
    uv run python -m preprocessing --report          # quality report only
    uv run python -m preprocessing --single who/who_63229.mp4
"""

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .runner import PreprocessingPipeline


def _progress(current: int, total: int, name: str, result: dict):
    status = result["status"]
    icon = {"ok": "✓", "skipped": "⊘", "error": "✗"}.get(status, "?")
    elapsed = f" ({result['elapsed']:.1f}s)" if "elapsed" in result else ""
    warn = ""
    if status == "ok" and result.get("report", {}).get("warnings"):
        warn = f" ⚠ {', '.join(result['report']['warnings'])}"
    if status == "error":
        warn = f" — {result.get('error', 'unknown')}"
    print(f"  [{current}/{total}] {icon} {name}{elapsed}{warn}")


def main():
    parser = argparse.ArgumentParser(description="WLASL Video Preprocessing Pipeline")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/raw_video_data"),
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessed"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip RAFT optical flow computation (much faster)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-process videos even if output already exists",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print quality report for existing preprocessed data and exit",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process a single video by relative path (e.g. who/who_63229.mp4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for RAFT model",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    cfg = PipelineConfig(
        dataset_dir=args.dataset_dir,
        labels_csv=args.dataset_dir / "labels.csv",
        output_dir=args.output_dir,
        normalized_video_dir=args.output_dir / "_normalized_videos",
        target_fps=args.target_fps,
        target_sequence_length=args.sequence_length,
        device=args.device,
    )

    pipeline = PreprocessingPipeline(cfg)

    if args.report:
        print("\n📊 Dataset Quality Report\n")
        report = pipeline.report()
        print(f"  Total samples : {report['total_samples']}")
        print(f"  Passed        : {report['passed']}")
        print(f"  Failed        : {report['failed_count']}")
        print("\n  Per-class counts:")
        for cls, count in sorted(report["class_counts"].items()):
            print(f"    {cls:15s} : {count}")
        if report["failed_samples"]:
            print("\n  ⚠ Failed samples:")
            for f in report["failed_samples"]:
                print(f"    {f['label']}/{f['video']} — {'; '.join(f['warnings'])}")
        print()
        return

    print("\n🔧 WLASL Preprocessing Pipeline")
    print(f"   Dataset : {cfg.dataset_dir}")
    print(f"   Output  : {cfg.output_dir}")
    print(f"   Device  : {cfg.resolve_device()}")
    print(f"   FPS     : {cfg.target_fps}")
    print(f"   Seq Len : {cfg.target_sequence_length}")
    print(f"   Flow    : {'skip' if args.skip_flow else cfg.flow_model}")
    print()

    if args.single:
        # Process one video
        parts = args.single.replace("\\", "/").split("/")
        label = parts[0]
        video_name = parts[-1]
        video_path = cfg.dataset_dir / args.single

        if not video_path.exists():
            print(f"  ✗ File not found: {video_path}")
            sys.exit(1)

        print(f"  Processing {args.single} …")
        result = pipeline.process_single_video(
            video_path,
            label,
            video_name,
            skip_existing=not args.reprocess,
            skip_flow=args.skip_flow,
        )
        _progress(1, 1, video_name, result)
    else:
        # Process full dataset
        summary = pipeline.run(
            skip_existing=not args.reprocess,
            skip_flow=args.skip_flow,
            progress_cb=_progress,
        )
        print(
            f"\n  Done — {summary['processed']} processed, "
            f"{summary['skipped']} skipped, {summary['errors']} errors\n"
        )


if __name__ == "__main__":
    main()
