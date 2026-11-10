from langchain_core.tools import tool
from .data import LAST as L, CURRENT as C

PLANS: list[dict] = []  # in-memory stand-in for a CRM / ticketing system


def _pct(a, b):
    return round((b - a) / a * 100, 1)


@tool
def get_performance_summary() -> dict:
    """Get the seller's last-month vs this-month metrics: sales, orders,
    cancellation rate, average selling price and out-of-stock product count."""
    return {
        "sales": {"last": L["sales"], "current": C["sales"], "change_pct": _pct(L["sales"], C["sales"]),
                  "change_inr": C["sales"] - L["sales"]},
        "orders": {"last": L["orders"], "current": C["orders"], "change_pct": _pct(L["orders"], C["orders"])},
        "cancellation_rate_pct": {"last": round(L["cancel_rate"] * 100), "current": round(C["cancel_rate"] * 100)},
        "average_selling_price": {"last": L["asp"], "current": C["asp"], "change_pct": _pct(L["asp"], C["asp"])},
        "out_of_stock_products": {"last": L["oos"], "current": C["oos"], "multiple": round(C["oos"] / L["oos"], 1)},
    }


@tool
def diagnose_drop() -> dict:
    """Get a ranked, pre-computed breakdown of WHY sales dropped, with rupee impact
    and root cause analysis. Use this to identify: INVENTORY issues, CANCELLATION issues,
    VISIBILITY issues, and PRICING issues. This does the analysis for you."""
    order_drop = L["orders"] - C["orders"]
    net_last = L["orders"] * (1 - L["cancel_rate"])
    net_now = C["orders"] * (1 - C["cancel_rate"])
    net_now_old_rate = C["orders"] * (1 - L["cancel_rate"])
    extra_cancelled = net_now_old_rate - net_now  # orders lost purely from the higher cancel rate
    
    return {
        "total_sales_lost_inr": L["sales"] - C["sales"],
        "sales_drop_percentage": round(_pct(L["sales"], C["sales"])),
        
        "root_cause_analysis": {
            "pricing_issue": {
                "is_cause": False,
                "detail": "Average selling price unchanged at Rs 500. Price is NOT the problem. This is a VOLUME issue."
            },
            "inventory_issue": {
                "is_cause": True,
                "severity": "HIGH",
                "detail": f"Out-of-stock products jumped from {L['oos']} to {C['oos']} items ({round(_pct(L['oos'], C['oos']))}% increase).",
                "impact": f"Lost approximately {order_drop} orders worth Rs {order_drop * C['asp']:,}.",
                "explanation": "Out-of-stock items lose search visibility on Snapdeal, reducing traffic and orders."
            },
            "cancellation_issue": {
                "is_cause": True,
                "severity": "MEDIUM",
                "detail": f"Cancellation rate increased from {round(L['cancel_rate']*100)}% to {round(C['cancel_rate']*100)}%.",
                "impact": f"Lost approximately {round(extra_cancelled)} completed orders worth Rs {round(extra_cancelled * C['asp']):,}.",
                "explanation": "High cancellations (stock issues or late dispatch) hurt seller ranking and customer trust."
            },
            "visibility_issue": {
                "is_cause": True,
                "severity": "HIGH",
                "detail": f"With {C['oos']} products out of stock, your catalog visibility on Snapdeal search is significantly reduced.",
                "explanation": "Unavailable products don't appear in search results, causing traffic loss."
            }
        },
        
        "ranked_findings": [
            {"rank": 1, "category": "INVENTORY + VISIBILITY",
             "issue": f"Out-of-stock products tripled from {L['oos']} to {C['oos']}",
             "impact": f"Lost {order_drop} orders (~Rs {order_drop * C['asp']:,})",
             "action_needed": "Identify which products are out and prioritize restocking bestsellers"},
            
            {"rank": 2, "category": "CANCELLATIONS",
             "issue": f"Cancellation rate rose from {round(L['cancel_rate']*100)}% to {round(C['cancel_rate']*100)}%",
             "impact": f"Lost {round(extra_cancelled)} completed orders (~Rs {round(extra_cancelled * C['asp']):,})",
             "action_needed": "Find cancellation reasons (stock or delivery) and fix root cause"},
        ],
        
        "net_completed_orders": {"last": round(net_last), "current": round(net_now), "change_pct": _pct(net_last, net_now)},
        
        "follow_up_questions_to_ask": [
            "Which specific products are out of stock? Are they your bestsellers?",
            "Why are orders being cancelled? Stock unavailable or delivery delays?",
            "Any recent changes to pricing, suppliers, or product listings?"
        ]
    }


@tool
def save_action_plan(issues: list[str], actions: list[str], follow_up_days: int = 3) -> dict:
    """Save the agreed action plan for the seller and schedule a follow-up call.
    Call this AFTER getting seller agreement on the recommendations.
    This creates the final summary that will be sent to the seller.

    Args:
        issues: List of 2-3 key issues identified (e.g., ["High out-of-stock on bestsellers", "Cancellation rate at 14%"])
        actions: List of 2-3 concrete actions the seller agreed to take (e.g., ["Restock top 3 bestsellers by tomorrow", "Fix dispatch time to under 24 hours"])
        follow_up_days: Days until the follow-up call (default: 3)
    """
    if len(actions) < 2:
        return {"error": "Please provide at least 2 actions for a complete plan"}
    
    plan = {
        "id": f"PLAN-{len(PLANS) + 1:03d}",
        "seller": "Rakesh",
        "date_created": "today",
        "key_issues": issues,
        "agreed_actions": actions,
        "follow_up_in_days": follow_up_days,
        "status": "created",
        "delivery_method": "WhatsApp summary + scheduled callback",
        "summary_message": f"Action plan created with {len(actions)} actions. Follow-up scheduled in {follow_up_days} days."
    }
    PLANS.append(plan)
    return plan


TOOLS = [get_performance_summary, diagnose_drop, save_action_plan]
