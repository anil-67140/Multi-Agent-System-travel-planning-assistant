"""Budget breakdown."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tripmate.domain.money import Money


class BudgetCategory(StrEnum):
    FLIGHTS = "flights"
    LODGING = "lodging"
    FOOD = "food"
    LOCAL_TRANSPORT = "local_transport"
    ACTIVITIES = "activities"
    BUFFER = "buffer"


class BudgetVerdict(StrEnum):
    WITHIN_BUDGET = "within_budget"
    TIGHT = "tight"  # fits, but with under 10% headroom
    OVER_BUDGET = "over_budget"
    NO_LIMIT = "no_limit"


class BudgetLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: BudgetCategory
    amount: Money
    basis: str = Field(max_length=300)
    is_estimate: bool = True


class BudgetBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    lines: list[BudgetLine]
    total: Money
    limit: Money | None = None
    verdict: BudgetVerdict = BudgetVerdict.NO_LIMIT
    remaining: Decimal | None = None  # may be negative
    utilization_pct: float | None = None
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> BudgetBreakdown:
        for line in self.lines:
            if line.amount.currency != self.currency:
                raise ValueError("every budget line must use the plan currency")
        if self.total.currency != self.currency:
            raise ValueError("total must use the plan currency")
        summed = sum((line.amount.amount for line in self.lines), Decimal(0))
        if summed != self.total.amount:
            raise ValueError(f"total {self.total.amount} does not equal sum of lines {summed}")
        return self

    def line(self, category: BudgetCategory) -> Money | None:
        for entry in self.lines:
            if entry.category == category:
                return entry.amount
        return None
