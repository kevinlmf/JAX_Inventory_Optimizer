"""
Capital Efficiency Analyzer
分析资本效率，优化资金使用
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CapitalEfficiencyMetrics:
    """资本效率指标"""
    inventory_turnover: float
    days_inventory_outstanding: float
    cash_conversion_cycle: float
    return_on_working_capital: float
    capital_productivity: float
    efficiency_score: float
    benchmarks: Dict[str, float]
    improvement_recommendations: List[Dict[str, Any]]


class CapitalEfficiencyAnalyzer:
    """
    资本效率分析器

    核心功能：
    1. 计算关键资本效率指标
    2. 与行业基准对比
    3. 识别改进机会
    4. 量化优化潜力
    """

    def __init__(
        self,
        industry_benchmarks: Optional[Dict[str, float]] = None
    ):
        """
        初始化资本效率分析器

        Args:
            industry_benchmarks: 行业基准指标
        """
        # 默认行业基准（零售业）
        self.industry_benchmarks = industry_benchmarks or {
            'inventory_turnover': 8.0,  # 年周转8次
            'days_inventory_outstanding': 45.0,  # 45天
            'cash_conversion_cycle': 30.0,  # 30天
            'return_on_working_capital': 0.25,  # 25%
            'capital_productivity': 3.0  # 每元资本产出3元收入
        }

    def analyze(
        self,
        inventory_value: float,
        annual_cogs: float,
        annual_revenue: float,
        days_payable_outstanding: float = 30.0,
        days_sales_outstanding: float = 45.0,
        working_capital: float = 100000.0
    ) -> CapitalEfficiencyMetrics:
        """
        分析资本效率

        Args:
            inventory_value: 当前库存价值
            annual_cogs: 年度销售成本 (Cost of Goods Sold)
            annual_revenue: 年度营收
            days_payable_outstanding: 应付账款天数
            days_sales_outstanding: 应收账款天数
            working_capital: 营运资本

        Returns:
            CapitalEfficiencyMetrics: 资本效率指标
        """
        # 计算库存周转率
        inventory_turnover = self._calculate_inventory_turnover(
            annual_cogs, inventory_value
        )

        # 计算库存周转天数
        days_inventory_outstanding = self._calculate_dio(inventory_turnover)

        # 计算现金转换周期
        cash_conversion_cycle = self._calculate_ccc(
            days_inventory_outstanding,
            days_sales_outstanding,
            days_payable_outstanding
        )

        # 计算营运资本回报率
        return_on_working_capital = self._calculate_rowc(
            annual_revenue, annual_cogs, working_capital
        )

        # 计算资本生产率
        capital_productivity = self._calculate_capital_productivity(
            annual_revenue, working_capital
        )

        # 计算综合效率得分
        efficiency_score = self._calculate_efficiency_score(
            inventory_turnover,
            cash_conversion_cycle,
            return_on_working_capital,
            capital_productivity
        )

        # 生成改进建议
        improvement_recommendations = self._generate_improvement_recommendations(
            inventory_turnover,
            days_inventory_outstanding,
            cash_conversion_cycle,
            return_on_working_capital,
            capital_productivity
        )

        return CapitalEfficiencyMetrics(
            inventory_turnover=float(inventory_turnover),
            days_inventory_outstanding=float(days_inventory_outstanding),
            cash_conversion_cycle=float(cash_conversion_cycle),
            return_on_working_capital=float(return_on_working_capital),
            capital_productivity=float(capital_productivity),
            efficiency_score=float(efficiency_score),
            benchmarks=self.industry_benchmarks,
            improvement_recommendations=improvement_recommendations
        )

    def _calculate_inventory_turnover(
        self,
        annual_cogs: float,
        avg_inventory_value: float
    ) -> float:
        """
        计算库存周转率
        Inventory Turnover = COGS / Average Inventory
        """
        if avg_inventory_value == 0:
            return 0.0
        return annual_cogs / avg_inventory_value

    def _calculate_dio(self, inventory_turnover: float) -> float:
        """
        计算库存周转天数 (Days Inventory Outstanding)
        DIO = 365 / Inventory Turnover
        """
        if inventory_turnover == 0:
            return 365.0
        return 365.0 / inventory_turnover

    def _calculate_ccc(
        self,
        dio: float,
        dso: float,
        dpo: float
    ) -> float:
        """
        计算现金转换周期 (Cash Conversion Cycle)
        CCC = DIO + DSO - DPO
        """
        return dio + dso - dpo

    def _calculate_rowc(
        self,
        annual_revenue: float,
        annual_cogs: float,
        working_capital: float
    ) -> float:
        """
        计算营运资本回报率 (Return on Working Capital)
        ROWC = (Revenue - COGS) / Working Capital
        """
        if working_capital == 0:
            return 0.0
        gross_profit = annual_revenue - annual_cogs
        return gross_profit / working_capital

    def _calculate_capital_productivity(
        self,
        annual_revenue: float,
        working_capital: float
    ) -> float:
        """
        计算资本生产率
        Capital Productivity = Revenue / Working Capital
        """
        if working_capital == 0:
            return 0.0
        return annual_revenue / working_capital

    def _calculate_efficiency_score(
        self,
        inventory_turnover: float,
        ccc: float,
        rowc: float,
        capital_productivity: float
    ) -> float:
        """
        计算综合效率得分 (0-100)

        基于各指标与行业基准的对比
        """
        # 库存周转率得分
        turnover_score = min(100, (inventory_turnover / self.industry_benchmarks['inventory_turnover']) * 100)

        # 现金转换周期得分（越低越好）
        ccc_score = min(100, (self.industry_benchmarks['cash_conversion_cycle'] / max(ccc, 1)) * 100)

        # 营运资本回报率得分
        rowc_score = min(100, (rowc / self.industry_benchmarks['return_on_working_capital']) * 100)

        # 资本生产率得分
        productivity_score = min(100, (capital_productivity / self.industry_benchmarks['capital_productivity']) * 100)

        # 加权平均
        weights = {
            'turnover': 0.3,
            'ccc': 0.3,
            'rowc': 0.2,
            'productivity': 0.2
        }

        total_score = (
            turnover_score * weights['turnover'] +
            ccc_score * weights['ccc'] +
            rowc_score * weights['rowc'] +
            productivity_score * weights['productivity']
        )

        return float(total_score)

    def _generate_improvement_recommendations(
        self,
        inventory_turnover: float,
        dio: float,
        ccc: float,
        rowc: float,
        capital_productivity: float
    ) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []

        # 库存周转率改进
        if inventory_turnover < self.industry_benchmarks['inventory_turnover']:
            gap = self.industry_benchmarks['inventory_turnover'] - inventory_turnover
            potential_savings = gap * 10000  # 假设每提高1次周转节省1万元

            recommendations.append({
                'area': 'INVENTORY_TURNOVER',
                'priority': 'HIGH',
                'current_value': float(inventory_turnover),
                'target_value': float(self.industry_benchmarks['inventory_turnover']),
                'gap': float(gap),
                'potential_savings': float(potential_savings),
                'actions': [
                    '优化采购频次，减少单次采购量',
                    '清理滞销库存，提高库存流动性',
                    '实施JIT订货策略'
                ]
            })

        # 现金转换周期改进
        if ccc > self.industry_benchmarks['cash_conversion_cycle']:
            gap = ccc - self.industry_benchmarks['cash_conversion_cycle']
            # 每减少1天CCC，释放的现金 = 日均COGS
            daily_cogs = 10000  # 简化假设
            potential_cash_release = gap * daily_cogs

            recommendations.append({
                'area': 'CASH_CONVERSION_CYCLE',
                'priority': 'HIGH',
                'current_value': float(ccc),
                'target_value': float(self.industry_benchmarks['cash_conversion_cycle']),
                'gap': float(gap),
                'potential_cash_release': float(potential_cash_release),
                'actions': [
                    f'缩短库存周转天数（当前{dio:.0f}天）',
                    '加速应收账款回收',
                    '协商延长应付账款账期'
                ]
            })

        # 营运资本回报率改进
        if rowc < self.industry_benchmarks['return_on_working_capital']:
            gap = self.industry_benchmarks['return_on_working_capital'] - rowc
            potential_profit_increase = gap * 100000  # 假设营运资本10万

            recommendations.append({
                'area': 'RETURN_ON_WORKING_CAPITAL',
                'priority': 'MEDIUM',
                'current_value': float(rowc),
                'target_value': float(self.industry_benchmarks['return_on_working_capital']),
                'gap': float(gap),
                'potential_profit_increase': float(potential_profit_increase),
                'actions': [
                    '提高产品毛利率',
                    '减少营运资本占用',
                    '优化产品组合，聚焦高利润产品'
                ]
            })

        # 资本生产率改进
        if capital_productivity < self.industry_benchmarks['capital_productivity']:
            gap = self.industry_benchmarks['capital_productivity'] - capital_productivity
            potential_revenue_increase = gap * 100000  # 假设营运资本10万

            recommendations.append({
                'area': 'CAPITAL_PRODUCTIVITY',
                'priority': 'MEDIUM',
                'current_value': float(capital_productivity),
                'target_value': float(self.industry_benchmarks['capital_productivity']),
                'gap': float(gap),
                'potential_revenue_increase': float(potential_revenue_increase),
                'actions': [
                    '提高销售效率',
                    '减少闲置库存',
                    '优化资金配置'
                ]
            })

        # 按优先级排序
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])

        return recommendations

    def compare_scenarios(
        self,
        scenarios: Dict[str, Dict[str, Any]]
    ) -> Dict[str, CapitalEfficiencyMetrics]:
        """
        比较不同场景下的资本效率

        Args:
            scenarios: 场景参数字典

        Returns:
            各场景的分析结果
        """
        results = {}

        for scenario_name, params in scenarios.items():
            metrics = self.analyze(
                inventory_value=params['inventory_value'],
                annual_cogs=params['annual_cogs'],
                annual_revenue=params['annual_revenue'],
                days_payable_outstanding=params.get('days_payable_outstanding', 30.0),
                days_sales_outstanding=params.get('days_sales_outstanding', 45.0),
                working_capital=params.get('working_capital', 100000.0)
            )
            results[scenario_name] = metrics

        return results

    def calculate_roi(
        self,
        current_metrics: CapitalEfficiencyMetrics,
        investment_amount: float,
        expected_improvement: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        计算优化投资的ROI

        Args:
            current_metrics: 当前指标
            investment_amount: 投资金额
            expected_improvement: 预期改进（百分比）

        Returns:
            ROI分析结果
        """
        # 计算改进后的指标
        improved_turnover = current_metrics.inventory_turnover * (1 + expected_improvement.get('turnover', 0))
        improved_ccc = current_metrics.cash_conversion_cycle * (1 - expected_improvement.get('ccc', 0))

        # 估算收益
        # 1. 库存成本节约
        turnover_improvement = improved_turnover - current_metrics.inventory_turnover
        inventory_savings = turnover_improvement * 10000  # 简化估算

        # 2. 现金流改善
        ccc_improvement = current_metrics.cash_conversion_cycle - improved_ccc
        cash_flow_benefit = ccc_improvement * 1000  # 每天1000元

        # 总收益
        total_annual_benefit = inventory_savings + cash_flow_benefit * 365

        # 投资回报率
        roi = (total_annual_benefit - investment_amount) / investment_amount
        payback_period = investment_amount / (total_annual_benefit / 12)  # 月

        return {
            'investment_amount': float(investment_amount),
            'annual_benefit': float(total_annual_benefit),
            'inventory_savings': float(inventory_savings),
            'cash_flow_benefit': float(cash_flow_benefit * 365),
            'roi': float(roi),
            'roi_percentage': float(roi * 100),
            'payback_period_months': float(payback_period),
            'npv_3_years': float(total_annual_benefit * 3 - investment_amount)
        }


# Alias for backward compatibility
CapitalAnalyzer = CapitalEfficiencyAnalyzer
