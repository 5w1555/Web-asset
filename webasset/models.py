"""Data contracts for digital-asset acquisition underwriting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonthlyFinancials:
    """One month of reported operating performance."""

    period: str
    revenue: float
    operating_expenses: float = 0.0


@dataclass
class Scenario:
    """Operating assumptions applied to the latest normalized cash flow."""

    name: str
    revenue_change: float = 0.0
    expense_change: float = 0.0
    exit_multiple: float = 0.0


@dataclass
class DealInput:
    """User-supplied facts and assumptions for one acquisition."""

    name: str
    asset_type: str
    asking_price: float
    financials: list[MonthlyFinancials]
    transaction_costs: float = 0.0
    working_capital: float = 0.0
    replacement_labor_monthly: float = 0.0
    target_annual_return: float = 0.25
    holding_years: int = 5
    evidence: dict[str, str] = field(default_factory=dict)
    scenarios: list[Scenario] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    annual_cash_flow: float
    payback_years: float | None
    cash_on_cash_roi: float
    maximum_offer: float
    terminal_value: float


@dataclass
class UnderwritingResult:
    name: str
    asset_type: str
    asking_price: float
    normalized_monthly_cash_flow: float
    normalized_annual_cash_flow: float
    annualized_roi: float
    payback_years: float | None
    npv_at_target_return: float
    irr: float | None
    maximum_offer: float
    evidence_score: float
    decision: str
    warnings: list[str]
    scenarios: list[ScenarioResult]
