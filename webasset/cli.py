from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .core_v2 import analyze, crawl, save_json, analysis_to_dict, CrawlData
from .underwriting import deal_from_dict, result_to_dict, underwrite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="webasset",
        description="Analyze internal link opportunities across real-world websites.",
        epilog="Examples:\n  webasset crawl https://example.com --limit 100\n  webasset analyze https://example.com -o results.json\n  webasset underwrite deal.json -o underwriting.json\n  webasset report results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website and save raw page data")
    crawl_parser.add_argument("url", help="Root URL to crawl")
    crawl_parser.add_argument("-o", "--output", default="crawl.json", help="Output file (default: crawl.json)")
    crawl_parser.add_argument("--limit", type=int, default=50, help="Max pages to crawl (default: 50)")
    crawl_parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds (default: 15.0)")

    analyze_parser = subparsers.add_parser("analyze", help="Crawl, curate, and generate recommendations")
    analyze_parser.add_argument("url", help="Root URL to analyze")
    analyze_parser.add_argument("-o", "--output", default="results.json", help="Output file (default: results.json)")
    analyze_parser.add_argument("--limit", type=int, default=50, help="Max pages to analyze (default: 50)")
    analyze_parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds (default: 15.0)")
    analyze_parser.add_argument("--save-crawl", help="Also save raw crawl (default: crawl.results.json)")

    underwrite_parser = subparsers.add_parser("underwrite", help="Underwrite one digital-asset acquisition from JSON")
    underwrite_parser.add_argument("input", help="Path to a deal JSON file")
    underwrite_parser.add_argument("-o", "--output", default="underwriting.json", help="Output file")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple acquisition deal packets from JSON")
    compare_parser.add_argument("input", help="Path to a JSON array or an object with a deals array")
    compare_parser.add_argument("-o", "--output", default="underwriting-comparison.json", help="Output file")

    report_parser = subparsers.add_parser("report", help="Print a JSON analysis as readable output")
    report_parser.add_argument("input", help="Path to results.json")
    report_parser.add_argument("--filter", choices=["all", "high", "medium"], default="high", help="Filter by confidence (default: high)")
    report_parser.add_argument("--limit", type=int, default=None, help="Limit output rows")
    
    return parser


def print_report(report: dict, confidence_filter: str = "high", limit: int = None) -> None:
    """Print JSON report as human-readable output."""
    stats = report.get("statistics", {})
    crawl_stats = stats.get("crawl", {})
    curation_stats = stats.get("curation", {})
    rec_stats = stats.get("recommendations", {})
    
    print(f"\n{'='*70}")
    print(f"Site: {report.get('root_url', 'unknown')}")
    print(f"Crawled: {crawl_stats.get('successful', 0)} pages | Analyzed: {curation_stats.get('usable_for_analysis', 0)} usable")
    print(f"Recommendations: {rec_stats.get('total', 0)} total | {rec_stats.get('high_confidence', 0)} HIGH")
    print(f"{'='*70}\n")
    
    recommendations = report.get("recommendations", [])
    
    if not recommendations:
        print("No recommendations found.")
        return
    
    print(f"{'Source':<40} {'Target':<40} {'Score':<8} {'Anchor':<25}")
    print(f"{'-'*40} {'-'*40} {'-'*8} {'-'*25}")
    
    shown = 0
    for rec in recommendations:
        score_percent = rec.get("score_percent", 0)
        if confidence_filter == "high" and score_percent < 45:
            continue
        if confidence_filter == "medium" and score_percent < 25:
            continue
        if limit and shown >= limit:
            break
        
        src = rec["source"][:35] + ".." if len(rec["source"]) > 37 else rec["source"]
        tgt = rec["target"][:35] + ".." if len(rec["target"]) > 37 else rec["target"]
        anchor = rec["anchor_text"][:23] + ".." if len(rec["anchor_text"]) > 25 else rec["anchor_text"]
        
        print(f"{src:<40} {tgt:<40} {rec['score_percent']:>6.0f}% {anchor:<25}")
        shown += 1
    
    if limit and shown >= limit:
        print(f"\n... and {len(recommendations) - shown} more recommendations")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "crawl":
            print(f"Crawling {args.url}...")
            crawl_data = crawl(args.url, limit=args.limit, timeout=args.timeout)
            save_json(asdict(crawl_data), args.output)
            stats = crawl_data.crawl_stats
            print(f"✓ Crawled: {stats.get('successful', 0)} OK | {stats.get('errors', 0)} errors | {stats.get('off_domain', 0)} off-domain")
            print(f"  Saved to {args.output}")
            
        elif args.command == "analyze":
            print(f"Analyzing {args.url}...")
            crawl_data = crawl(args.url, limit=args.limit, timeout=args.timeout)
            result = analyze(crawl_data)
            
            # Save analysis
            analysis_dict = analysis_to_dict(result)
            save_json(analysis_dict, args.output)
            
            # Optionally save raw crawl
            if args.save_crawl:
                crawl_dict = asdict(result.crawl_data)
                save_json(crawl_dict, args.save_crawl)
            
            # Print summary
            print_report(analysis_dict, confidence_filter="high")
            print(f"\nSaved to {args.output}")

        elif args.command == "underwrite":
            with open(args.input, encoding="utf-8") as f:
                deal = deal_from_dict(json.load(f))
            result = result_to_dict(underwrite(deal))
            save_json(result, args.output)
            print(f"Decision: {result['decision']} | Maximum offer: {result['maximum_offer']:.2f}")
            print(f"Annual cash flow: {result['normalized_annual_cash_flow']:.2f} | Evidence score: {result['evidence_score']:.0%}")
            irr = "n/a" if result["irr"] is None else f"{result['irr']:.1%}"
            print(f"Payback: {result['payback_years'] or 'n/a'} years | IRR: {irr} | NPV: {result['npv_at_target_return']:.2f}")
            for warning in result["warnings"]:
                print(f"Warning: {warning}")
            print(f"Saved to {args.output}")

        elif args.command == "compare":
            with open(args.input, encoding="utf-8") as f:
                payload = json.load(f)
            deal_items = payload.get("deals", []) if isinstance(payload, dict) else payload
            results = [result_to_dict(underwrite(deal_from_dict(item))) for item in deal_items]
            results.sort(key=lambda item: item["maximum_offer"], reverse=True)
            save_json({"results": results}, args.output)
            for result in results:
                print(f"{result['decision']:<10} {result['name']:<30} max offer {result['maximum_offer']:.2f}")
            print(f"Saved to {args.output}")
            
        elif args.command == "report":
            with open(args.input, encoding="utf-8") as f:
                report = json.load(f)
            print_report(report, confidence_filter=args.filter, limit=args.limit)
            
        return 0
        
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"webasset: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
