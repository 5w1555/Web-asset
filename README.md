# WebAsset

**WebAsset is a local-first investment desk for traffic-producing digital assets.** It helps you price a domain or small website, identify what must be verified before you buy it, model a lightweight renovation, and compare opportunities on downside cash flow—not on a seller's headline multiple.

It is built for the common workflow: buy a domain or neglected site, add useful content, distribution, lead capture, or monetization, then own a durable stream of traffic and cash flow. It is decision support, not financial, tax, or legal advice.

## What it answers

- **What is the highest purchase price that still meets my return hurdle?** `maximum_offer` deducts transaction costs, working capital, and the one-time renovation budget.
- **Will the improvement work pay for itself?** Provide launch time, build cost, ongoing cost, and expected revenue/traffic lift to calculate renovation payback.
- **Is the traffic worth owning?** Traffic quality rewards organic/direct demand, diversification, and a stable six-month trend; it flags one-source dependency.
- **What can break the deal?** Evidence gates prevent an unverified deal from being a `BUY`; high trademark risk is an automatic `PASS`.
- **Which deal deserves attention first?** Compare deal packets, sorted by downside maximum offer.

## Install

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Underwrite a domain or site

Create `domain-deal.json`:

```json
{
  "name": "BicycleRepair.com",
  "asset_type": "domain_with_content",
  "asking_price": 18000,
  "transaction_costs": 500,
  "working_capital": 1000,
  "target_annual_return": 0.30,
  "holding_years": 5,
  "financials": [
    {"period": "2026-07", "revenue": 1800, "operating_expenses": 300},
    {"period": "2026-08", "revenue": 1900, "operating_expenses": 300}
  ],
  "traffic": {
    "monthly_visits": 24000,
    "organic_share": 0.62,
    "direct_share": 0.12,
    "top_source_share": 0.43,
    "six_month_trend": 0.08
  },
  "renovation": {
    "one_time_cost": 3500,
    "monthly_cost": 250,
    "launch_months": 2,
    "monthly_revenue_uplift": 900,
    "expected_monthly_visits_uplift": 5000
  },
  "revenue_concentration": 0.35,
  "trademark_risk": "low",
  "evidence": {
    "financials": "verified",
    "analytics": "verified",
    "bank_statements": "verified",
    "transferability": "verified"
  },
  "scenarios": [
    {"name": "downside", "traffic_change": -0.25, "revenue_per_visit_change": -0.10, "exit_multiple": 1},
    {"name": "base", "exit_multiple": 2},
    {"name": "upside", "traffic_change": 0.20, "revenue_change": 0.10, "exit_multiple": 3}
  ]
}
```

```shell
webasset underwrite domain-deal.json -o underwriting.json
webasset compare deal-packets.json -o comparison.json
```

All shares and changes are decimal values: `0.25` means 25%; `-0.10` means a 10% decline. Scenario traffic and revenue-per-visit changes compound before the scenario revenue adjustment. A missing traffic baseline does not fabricate a score: it produces a warning and a zero traffic-quality score.

### Decision rules

| Result | Meaning |
| --- | --- |
| `BUY` | Every required evidence item is verified; the downside case clears the return hurdle and price cap. |
| `NEGOTIATE` | Some evidence is incomplete, or the price exceeds the downside maximum offer. |
| `PASS` | No evidence, a failed downside hurdle, or high/confirmed trademark risk. |

Required evidence is `financials`, `analytics`, `bank_statements`, and `transferability`. The model intentionally uses only supplied values; it does not verify seller claims or appraise a domain automatically.

## Existing site research tools

The crawler and analysis pipeline are retained for understanding an asset after discovery or during diligence:

```shell
webasset crawl https://example.com --limit 100 -o crawl.json
webasset analyze https://example.com --limit 100 -o results.json
webasset report results.json --filter high --limit 20
```

They preserve raw observations and generate inspectable internal-link opportunities, useful when a renovation plan includes content consolidation or internal distribution.

## Development

```shell
python -m unittest discover -s tests -v
python -m compileall webasset
python -m webasset --help
```
