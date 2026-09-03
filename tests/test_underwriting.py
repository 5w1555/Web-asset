import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from webasset.cli import main, print_report
from webasset.underwriting import deal_from_dict, underwrite


class UnderwritingTests(unittest.TestCase):
    def setUp(self):
        self.deal = deal_from_dict({
            "name": "Example Content Site",
            "asset_type": "content",
            "asking_price": 30000,
            "financials": [
                {"period": "2026-01", "revenue": 6000, "operating_expenses": 2000},
                {"period": "2026-02", "revenue": 6000, "operating_expenses": 2000},
            ],
            "evidence": {
                "financials": "verified",
                "analytics": "verified",
                "bank_statements": "verified",
                "transferability": "verified",
            },
            "scenarios": [
                {"name": "downside", "traffic_change": -0.25, "revenue_change": -0.10, "exit_multiple": 1},
                {"name": "base", "exit_multiple": 2},
            ],
        })

    def test_normalizes_cash_flow_and_orders_scenarios(self):
        result = underwrite(self.deal)
        self.assertEqual(result.normalized_monthly_cash_flow, 4000)
        self.assertEqual(result.normalized_annual_cash_flow, 48000)
        self.assertEqual(result.decision, "BUY")
        self.assertIsNotNone(result.irr)
        self.assertGreater(result.npv_at_target_return, 0)
        self.assertLess(result.scenarios[0].annual_cash_flow, result.scenarios[1].annual_cash_flow)

    def test_renovation_and_traffic_are_included_in_price_and_risk(self):
        self.deal.traffic.monthly_visits = 10_000
        self.deal.traffic.organic_share = 0.70
        self.deal.traffic.direct_share = 0.15
        self.deal.traffic.top_source_share = 0.35
        self.deal.traffic.six_month_trend = 0.10
        self.deal.renovation.one_time_cost = 2_000
        self.deal.renovation.monthly_cost = 100
        self.deal.renovation.monthly_revenue_uplift = 600
        result = underwrite(self.deal)
        self.assertEqual(result.total_investment, 32_000)
        self.assertEqual(result.renovation_payback_months, 4)
        self.assertGreater(result.traffic_quality_score, 0.5)
        self.assertEqual(result.scenarios[0].annual_visits, 90_000)

    def test_improvement_actions_explain_how_to_build_value(self):
        self.deal.traffic.monthly_visits = 10_000
        self.deal.traffic.organic_share = 0.20
        self.deal.traffic.direct_share = 0.05
        self.deal.traffic.top_source_share = 0.70
        self.deal.traffic.six_month_trend = -0.10
        self.deal.revenue_concentration = 0.60
        result = underwrite(self.deal)
        plan = " ".join(result.improvement_actions).lower()
        self.assertIn("organic", plan)
        self.assertIn("email", plan)
        self.assertIn("diversify acquisition", plan)
        self.assertIn("traffic decline", plan)
        self.assertIn("diversify revenue", plan)

    def test_base_valuation_includes_renovation_cash_flow(self):
        self.deal.renovation.monthly_cost = 100
        self.deal.renovation.monthly_revenue_uplift = 600
        result = underwrite(self.deal)
        self.assertEqual(result.scenarios[1].annual_cash_flow, 54_000)
        self.assertGreater(result.npv_at_target_return, 0)

    def test_high_trademark_risk_forces_pass(self):
        self.deal.trademark_risk = "high"
        result = underwrite(self.deal)
        self.assertEqual(result.decision, "PASS")
        self.assertTrue(any("trademark" in warning for warning in result.warnings))

    def test_unverified_evidence_cannot_produce_buy(self):
        self.deal.evidence = {}
        result = underwrite(self.deal)
        self.assertEqual(result.decision, "PASS")
        self.assertTrue(result.warnings)

    def test_report_filter_excludes_low_scores(self):
        report = {
            "recommendations": [
                {"source": "a", "target": "b", "score_percent": 80, "anchor_text": "good"},
                {"source": "c", "target": "d", "score_percent": 20, "anchor_text": "weak"},
            ]
        }
        output = io.StringIO()
        with redirect_stdout(output):
            print_report(report, confidence_filter="high")
        self.assertIn("good", output.getvalue())
        self.assertNotIn("weak", output.getvalue())

    def test_cli_underwrite_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "deal.json"
            output_path = Path(directory) / "result.json"
            input_path.write_text(json.dumps({
                "name": "CLI Deal",
                "asking_price": 1000,
                "financials": [{"period": "2026-01", "revenue": 1000, "operating_expenses": 100}],
            }))
            self.assertEqual(main(["underwrite", str(input_path), "-o", str(output_path)]), 0)
            self.assertEqual(json.loads(output_path.read_text())["decision"], "PASS")


if __name__ == "__main__":
    unittest.main()
