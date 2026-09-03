"""Deterministic, inspectable acquisition underwriting calculations."""

from __future__ import annotations

from dataclasses import asdict

from .models import (
    DealInput,
    MonthlyFinancials,
    RenovationPlan,
    Scenario,
    ScenarioResult,
    TrafficMetrics,
    UnderwritingResult,
)


REQUIRED_EVIDENCE = {"financials", "analytics", "bank_statements", "transferability"}


def deal_from_dict(data: dict) -> DealInput:
    """Build a deal from the JSON input contract."""
    financials = [MonthlyFinancials(**item) for item in data.get("financials", [])]
    scenarios = [Scenario(**item) for item in data.get("scenarios", [])]
    traffic = TrafficMetrics(**data.get("traffic", {}))
    renovation = RenovationPlan(**data.get("renovation", {}))
    return DealInput(
        name=data["name"],
        asset_type=data.get("asset_type", "unknown"),
        asking_price=float(data["asking_price"]),
        financials=financials,
        transaction_costs=float(data.get("transaction_costs", 0.0)),
        working_capital=float(data.get("working_capital", 0.0)),
        replacement_labor_monthly=float(data.get("replacement_labor_monthly", 0.0)),
        target_annual_return=float(data.get("target_annual_return", 0.25)),
        holding_years=int(data.get("holding_years", 5)),
        evidence=data.get("evidence", {}),
        scenarios=scenarios,
        traffic=traffic,
        renovation=renovation,
        revenue_concentration=float(data.get("revenue_concentration", 0.0)),
        trademark_risk=str(data.get("trademark_risk", "unknown")),
    )


def _monthly_revenue_expenses(deal: DealInput) -> tuple[float, float]:
    if not deal.financials:
        raise ValueError("at least one monthly financial period is required")
    revenue = sum(item.revenue for item in deal.financials) / len(deal.financials)
    expenses = sum(item.operating_expenses for item in deal.financials) / len(deal.financials)
    return revenue, expenses


def _maximum_offer(annual_cash_flow: float, target_return: float, years: int, exit_multiple: float) -> float:
    if annual_cash_flow <= 0 or target_return <= 0 or years <= 0:
        return 0.0
    annuity_factor = (1 - (1 + target_return) ** -years) / target_return
    terminal_value = annual_cash_flow * max(0.0, exit_multiple)
    return max(0.0, annual_cash_flow * annuity_factor + terminal_value / (1 + target_return) ** years)


def _npv(annual_cash_flow: float, discount_rate: float, years: int, terminal_value: float, initial_cost: float) -> float:
    if years <= 0 or discount_rate <= -1:
        return -initial_cost
    return sum(annual_cash_flow / (1 + discount_rate) ** year for year in range(1, years + 1)) + terminal_value / (1 + discount_rate) ** years - initial_cost


def _irr(annual_cash_flow: float, years: int, terminal_value: float, initial_cost: float) -> float | None:
    if years <= 0 or initial_cost <= 0 or annual_cash_flow <= 0:
        return None
    low, high = -0.99, 10.0
    if _npv(annual_cash_flow, high, years, terminal_value, initial_cost) > 0:
        return None
    for _ in range(100):
        midpoint = (low + high) / 2
        if _npv(annual_cash_flow, midpoint, years, terminal_value, initial_cost) > 0:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _evidence_score(evidence: dict[str, str]) -> tuple[float, list[str]]:
    verified = sum(value.lower() == "verified" for value in evidence.values())
    score = verified / len(REQUIRED_EVIDENCE) if REQUIRED_EVIDENCE else 0.0
    warnings = [f"missing or unverified evidence: {key}" for key in sorted(REQUIRED_EVIDENCE) if evidence.get(key, "").lower() != "verified"]
    return min(1.0, score), warnings


def _traffic_quality_score(traffic: TrafficMetrics) -> float:
    """Score durable, diversified acquisition traffic without inventing precision."""
    if traffic.monthly_visits <= 0:
        return 0.0
    organic = min(1.0, max(0.0, traffic.organic_share))
    direct = min(1.0, max(0.0, traffic.direct_share))
    diversification = 1 - min(1.0, max(0.0, traffic.top_source_share))
    trend = min(1.0, max(0.0, (traffic.six_month_trend + 0.30) / 0.60))
    return round(0.35 * organic + 0.20 * direct + 0.30 * diversification + 0.15 * trend, 3)


def _risk_warnings(deal: DealInput) -> list[str]:
    warnings: list[str] = []
    if deal.traffic.monthly_visits <= 0:
        warnings.append("no traffic baseline supplied; traffic durability cannot be assessed")
    elif deal.traffic.top_source_share > 0.60:
        warnings.append("traffic concentration is high: one source supplies over 60% of visits")
    if deal.traffic.six_month_trend < -0.15:
        warnings.append("traffic has declined more than 15% over six months")
    if deal.revenue_concentration > 0.50:
        warnings.append("revenue concentration is high: one source supplies over 50% of revenue")
    if deal.trademark_risk.lower() in {"high", "confirmed"}:
        warnings.append("high trademark risk requires legal clearance before acquisition")
    if deal.renovation.launch_months > 6:
        warnings.append("renovation plan delays launch by more than six months")
    return warnings


def underwrite(deal: DealInput) -> UnderwritingResult:
    """Underwrite a deal using cash flow, return, evidence, and scenarios."""
    monthly_revenue, monthly_expenses = _monthly_revenue_expenses(deal)
    monthly_cash_flow = monthly_revenue - monthly_expenses - deal.replacement_labor_monthly
    annual_cash_flow = monthly_cash_flow * 12
    renovation = deal.renovation
    total_cost = deal.asking_price + deal.transaction_costs + deal.working_capital + renovation.one_time_cost
    roi = annual_cash_flow / total_cost if total_cost > 0 else 0.0
    payback = total_cost / annual_cash_flow if annual_cash_flow > 0 else None
    evidence_score, warnings = _evidence_score(deal.evidence)
    warnings.extend(_risk_warnings(deal))
    traffic_quality_score = _traffic_quality_score(deal.traffic)
    incremental_monthly_profit = renovation.monthly_revenue_uplift - renovation.monthly_cost
    renovation_payback = (
        renovation.launch_months + renovation.one_time_cost / incremental_monthly_profit
        if incremental_monthly_profit > 0 and renovation.one_time_cost > 0
        else None
    )

    scenarios = deal.scenarios or [Scenario(name="base", exit_multiple=0.0)]
    scenario_results = []
    for scenario in scenarios:
        traffic_factor = (1 + scenario.traffic_change) * (1 + scenario.revenue_per_visit_change)
        scenario_revenue = (monthly_revenue * traffic_factor + renovation.monthly_revenue_uplift) * 12 * (1 + scenario.revenue_change)
        scenario_expenses = (monthly_expenses + renovation.monthly_cost) * 12 * (1 + scenario.expense_change)
        scenario_cash_flow = scenario_revenue - scenario_expenses - deal.replacement_labor_monthly * 12
        gross_maximum = _maximum_offer(scenario_cash_flow, deal.target_annual_return, deal.holding_years, scenario.exit_multiple)
        scenario_results.append(ScenarioResult(
            name=scenario.name,
            annual_cash_flow=scenario_cash_flow,
            payback_years=total_cost / scenario_cash_flow if scenario_cash_flow > 0 else None,
            cash_on_cash_roi=scenario_cash_flow / total_cost if total_cost > 0 else 0.0,
            maximum_offer=max(0.0, gross_maximum - deal.transaction_costs - deal.working_capital - renovation.one_time_cost),
            terminal_value=scenario_cash_flow * max(0.0, scenario.exit_multiple),
            annual_visits=deal.traffic.monthly_visits * 12 * (1 + scenario.traffic_change) + renovation.expected_monthly_visits_uplift * 12,
        ))

    downside = next((item for item in scenario_results if item.name.lower() == "downside"), scenario_results[0])
    base_scenario = next((item for item in scenario_results if item.name.lower() == "base"), scenario_results[0])
    base_exit_value = base_scenario.terminal_value
    npv_at_target_return = _npv(annual_cash_flow, deal.target_annual_return, deal.holding_years, base_exit_value, total_cost)
    irr = _irr(annual_cash_flow, deal.holding_years, base_exit_value, total_cost)

    fatal_risk = deal.trademark_risk.lower() in {"high", "confirmed"}
    if fatal_risk:
        decision = "PASS"
    elif evidence_score < 1.0:
        decision = "PASS" if evidence_score == 0.0 else "NEGOTIATE"
    elif downside.cash_on_cash_roi < deal.target_annual_return:
        decision = "PASS"
    elif deal.asking_price > downside.maximum_offer:
        decision = "NEGOTIATE"
    else:
        decision = "BUY"

    return UnderwritingResult(
        name=deal.name,
        asset_type=deal.asset_type,
        asking_price=deal.asking_price,
        normalized_monthly_cash_flow=monthly_cash_flow,
        normalized_annual_cash_flow=annual_cash_flow,
        annualized_roi=roi,
        payback_years=payback,
        npv_at_target_return=npv_at_target_return,
        irr=irr,
        maximum_offer=downside.maximum_offer,
        total_investment=total_cost,
        traffic_quality_score=traffic_quality_score,
        renovation_payback_months=renovation_payback,
        evidence_score=evidence_score,
        decision=decision,
        warnings=warnings,
        scenarios=scenario_results,
    )


def result_to_dict(result: UnderwritingResult) -> dict:
    return asdict(result)
