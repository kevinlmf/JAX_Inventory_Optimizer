"""
Cost Optimization System - 企业三大核心成本优化

聚焦企业最关心的三大成本：
1. 沟通成本 - AI自动决策，减少人工干预
2. 库存成本 - 动态最优库存，零库存运营
3. 现金流成本 - 资金占用优化，风险预警

所有优化都以ROI为导向，可量化、可追踪
"""

from .inventory_optimizer import DynamicInventoryOptimizer
from .deadstock_detector import DeadstockDetector
from .jit_optimizer import JITOrderingOptimizer
from .cashflow_predictor import CashFlowPredictor
from .capital_analyzer import CapitalEfficiencyAnalyzer
from .auto_decision_engine import AutoDecisionEngine

__all__ = [
    'DynamicInventoryOptimizer',
    'DeadstockDetector',
    'JITOrderingOptimizer',
    'CashFlowPredictor',
    'CapitalEfficiencyAnalyzer',
    'AutoDecisionEngine'
]
