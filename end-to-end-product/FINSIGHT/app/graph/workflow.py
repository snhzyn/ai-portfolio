"""
LangGraph workflow definition for FINSIGHT.

This module wires together the full multi-agent workflow:

Manager
  ↓
Parallel Workers
  ├─ Macro
  ├─ Markets
  ├─ Commodities & FX
  └─ Geopolitical
  ↓
Lead Analyst
  ↓
Report

Simple test version uses dummy node implementations so the graph can be tested
before real data retrieval and Claude integration are added.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    commodities_fx_worker_node,
    geopolitical_worker_node,
    lead_analyst_node,
    macro_worker_node,
    manager_node,
    markets_worker_node,
    report_node,
)
from app.graph.state import FinanceBriefingState


def build_workflow():
    """
    Build and compile the FINSIGHT LangGraph workflow.
    """
    
    graph = StateGraph(FinanceBriefingState)

    graph.add_node("manager", manager_node)
    graph.add_node("macro_worker", macro_worker_node)
    graph.add_node("markets_worker", markets_worker_node)
    graph.add_node("commodities_fx_worker", commodities_fx_worker_node)
    graph.add_node("geopolitical_worker", geopolitical_worker_node)
    graph.add_node("lead_analyst", lead_analyst_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "manager")

    graph.add_edge("manager", "macro_worker")
    graph.add_edge("manager", "markets_worker")
    graph.add_edge("manager", "commodities_fx_worker")
    graph.add_edge("manager", "geopolitical_worker")

    graph.add_edge("macro_worker", "lead_analyst")
    graph.add_edge("markets_worker", "lead_analyst")
    graph.add_edge("commodities_fx_worker", "lead_analyst")
    graph.add_edge("geopolitical_worker", "lead_analyst")

    graph.add_edge("lead_analyst", "report")
    graph.add_edge("report", END)

    return graph.compile()