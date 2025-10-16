"""
Cash Flow Predictor
预测现金流，优化资金占用
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CashFlowPrediction:
    """现金流预测结果"""
    forecast_dates: List[str]
    inflow_forecast: jnp.ndarray
    outflow_forecast: jnp.ndarray
    net_cashflow: jnp.ndarray
    cumulative_cashflow: jnp.ndarray
    risk_periods: List[Dict[str, Any]]
    working_capital_needs: float
    recommendations: Dict[str, Any]


class CashFlowPredictor:
    """
    现金流预测器

    核心功能：
    1. 预测未来现金流入/流出
    2. 识别资金紧张期
    3. 优化营运资本需求
    4. 提供融资建议
    """

    def __init__(
        self,
        safety_buffer: float = 0.15,
        planning_horizon_days: int = 90
    ):
        """
        初始化现金流预测器

        Args:
            safety_buffer: 安全缓冲比例 (15%为默认)
            planning_horizon_days: 规划周期（天）
        """
        self.safety_buffer = safety_buffer
        self.planning_horizon_days = planning_horizon_days

    def predict(
        self,
        inventory_plan: jnp.ndarray,
        unit_costs: jnp.ndarray,
        sales_forecast: jnp.ndarray,
        selling_prices: jnp.ndarray,
        payment_terms_days: int = 30,
        collection_days: int = 45,
        current_cash: float = 100000.0
    ) -> CashFlowPrediction:
        """
        预测现金流

        Args:
            inventory_plan: 未来采购计划（数量）
            unit_costs: 单位采购成本
            sales_forecast: 销售预测（数量）
            selling_prices: 销售价格
            payment_terms_days: 付款账期（天）
            collection_days: 收款账期（天）
            current_cash: 当前现金余额

        Returns:
            CashFlowPrediction: 现金流预测结果
        """
        # 生成预测日期
        forecast_dates = self._generate_forecast_dates(len(sales_forecast))

        # 计算现金流出（采购付款）
        outflow = self._calculate_outflow(
            inventory_plan, unit_costs, payment_terms_days
        )

        # 计算现金流入（销售回款）
        inflow = self._calculate_inflow(
            sales_forecast, selling_prices, collection_days
        )

        # 计算净现金流和累计现金流
        net_cashflow = inflow - outflow
        cumulative_cashflow = jnp.cumsum(net_cashflow) + current_cash

        # 识别风险期
        risk_periods = self._identify_risk_periods(
            forecast_dates, cumulative_cashflow, current_cash
        )

        # 计算营运资本需求
        working_capital_needs = self._calculate_working_capital_needs(
            outflow, inflow, self.safety_buffer
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            cumulative_cashflow,
            risk_periods,
            working_capital_needs,
            current_cash
        )

        return CashFlowPrediction(
            forecast_dates=forecast_dates,
            inflow_forecast=inflow,
            outflow_forecast=outflow,
            net_cashflow=net_cashflow,
            cumulative_cashflow=cumulative_cashflow,
            risk_periods=risk_periods,
            working_capital_needs=float(working_capital_needs),
            recommendations=recommendations
        )

    def _generate_forecast_dates(self, num_periods: int) -> List[str]:
        """生成预测日期列表"""
        base_date = datetime.now()
        dates = []
        for i in range(num_periods):
            date = base_date + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
        return dates

    def _calculate_outflow(
        self,
        inventory_plan: jnp.ndarray,
        unit_costs: jnp.ndarray,
        payment_terms_days: int
    ) -> jnp.ndarray:
        """
        计算现金流出（考虑账期）

        采购在payment_terms_days后付款
        """
        # 计算采购金额
        purchase_amounts = inventory_plan * unit_costs

        # 延迟付款
        outflow = jnp.zeros_like(purchase_amounts)
        if payment_terms_days < len(purchase_amounts):
            outflow = jnp.concatenate([
                jnp.zeros(payment_terms_days),
                purchase_amounts[:-payment_terms_days]
            ])
        else:
            # 如果账期超过预测期，则在预测期内无需付款
            outflow = jnp.zeros_like(purchase_amounts)

        return outflow

    def _calculate_inflow(
        self,
        sales_forecast: jnp.ndarray,
        selling_prices: jnp.ndarray,
        collection_days: int
    ) -> jnp.ndarray:
        """
        计算现金流入（考虑账期）

        销售在collection_days后收款
        """
        # 计算销售金额
        sales_amounts = sales_forecast * selling_prices

        # 延迟收款
        inflow = jnp.zeros_like(sales_amounts)
        if collection_days < len(sales_amounts):
            inflow = jnp.concatenate([
                jnp.zeros(collection_days),
                sales_amounts[:-collection_days]
            ])
        else:
            inflow = jnp.zeros_like(sales_amounts)

        return inflow

    def _identify_risk_periods(
        self,
        forecast_dates: List[str],
        cumulative_cashflow: jnp.ndarray,
        current_cash: float
    ) -> List[Dict[str, Any]]:
        """识别现金流风险期"""
        risk_periods = []

        # 找出现金流低于安全水平的时期
        safety_level = current_cash * (1 - self.safety_buffer)

        for i, (date, cash) in enumerate(zip(forecast_dates, cumulative_cashflow)):
            if cash < safety_level:
                risk_level = 'HIGH' if cash < safety_level * 0.5 else 'MEDIUM'
                risk_periods.append({
                    'date': date,
                    'day_index': i,
                    'cash_balance': float(cash),
                    'deficit': float(safety_level - cash),
                    'risk_level': risk_level
                })

        return risk_periods

    def _calculate_working_capital_needs(
        self,
        outflow: jnp.ndarray,
        inflow: jnp.ndarray,
        safety_buffer: float
    ) -> float:
        """
        计算营运资本需求

        基于现金缺口的最大值
        """
        # 计算每日净现金流
        net_flow = inflow - outflow

        # 计算累计现金缺口
        cumulative_gap = jnp.cumsum(net_flow)

        # 找到最大缺口（最负值）
        max_deficit = jnp.abs(jnp.min(cumulative_gap))

        # 加上安全缓冲
        working_capital = max_deficit * (1 + safety_buffer)

        return float(working_capital)

    def _generate_recommendations(
        self,
        cumulative_cashflow: jnp.ndarray,
        risk_periods: List[Dict[str, Any]],
        working_capital_needs: float,
        current_cash: float
    ) -> Dict[str, Any]:
        """生成资金管理建议"""
        recommendations = {
            'status': 'HEALTHY',
            'actions': [],
            'financing_needed': False,
            'financing_amount': 0.0,
            'optimization_opportunities': []
        }

        # 检查是否需要融资
        if len(risk_periods) > 0:
            recommendations['status'] = 'AT_RISK'
            recommendations['financing_needed'] = True

            max_deficit = max([p['deficit'] for p in risk_periods])
            recommendations['financing_amount'] = float(max_deficit * 1.2)  # 加20%缓冲

            recommendations['actions'].append({
                'action': 'SECURE_FINANCING',
                'urgency': 'HIGH' if risk_periods[0]['risk_level'] == 'HIGH' else 'MEDIUM',
                'amount': recommendations['financing_amount'],
                'reason': f'预计在{len(risk_periods)}个时期出现资金紧张'
            })

        # 检查营运资本是否充足
        if working_capital_needs > current_cash:
            recommendations['actions'].append({
                'action': 'INCREASE_WORKING_CAPITAL',
                'urgency': 'MEDIUM',
                'amount': float(working_capital_needs - current_cash),
                'reason': '营运资本不足以支持计划运营'
            })

        # 优化机会
        if len(risk_periods) == 0:
            # 现金充足，可以考虑优化
            avg_cash = float(jnp.mean(cumulative_cashflow))
            if avg_cash > current_cash * 1.5:
                recommendations['optimization_opportunities'].append({
                    'opportunity': 'INVEST_EXCESS_CASH',
                    'description': '现金充裕，可考虑短期投资或提前付款获取折扣',
                    'estimated_benefit': float(avg_cash * 0.03)  # 假设3%年化收益
                })

        # 账期优化建议
        recommendations['optimization_opportunities'].append({
            'opportunity': 'OPTIMIZE_PAYMENT_TERMS',
            'description': '协商延长应付账款账期或缩短应收账款账期',
            'estimated_benefit': float(working_capital_needs * 0.1)
        })

        return recommendations

    def simulate_scenarios(
        self,
        base_inventory_plan: jnp.ndarray,
        unit_costs: jnp.ndarray,
        sales_forecast: jnp.ndarray,
        selling_prices: jnp.ndarray,
        scenarios: Dict[str, Dict[str, Any]]
    ) -> Dict[str, CashFlowPrediction]:
        """
        模拟不同场景下的现金流

        Args:
            base_inventory_plan: 基准采购计划
            unit_costs: 单位成本
            sales_forecast: 销售预测
            selling_prices: 销售价格
            scenarios: 场景参数字典

        Returns:
            Dict[str, CashFlowPrediction]: 各场景的预测结果
        """
        results = {}

        for scenario_name, params in scenarios.items():
            # 调整参数
            inventory_plan = base_inventory_plan * params.get('inventory_multiplier', 1.0)
            sales = sales_forecast * params.get('sales_multiplier', 1.0)

            prediction = self.predict(
                inventory_plan=inventory_plan,
                unit_costs=unit_costs,
                sales_forecast=sales,
                selling_prices=selling_prices,
                payment_terms_days=params.get('payment_terms_days', 30),
                collection_days=params.get('collection_days', 45),
                current_cash=params.get('current_cash', 100000.0)
            )

            results[scenario_name] = prediction

        return results

    def optimize_payment_terms(
        self,
        inventory_plan: jnp.ndarray,
        unit_costs: jnp.ndarray,
        sales_forecast: jnp.ndarray,
        selling_prices: jnp.ndarray,
        current_payment_terms: int,
        current_collection_days: int,
        target_working_capital: float
    ) -> Dict[str, Any]:
        """
        优化付款和收款账期

        Args:
            target_working_capital: 目标营运资本

        Returns:
            优化后的账期建议
        """
        best_terms = {
            'payment_terms_days': current_payment_terms,
            'collection_days': current_collection_days,
            'working_capital_needs': float('inf')
        }

        # 尝试不同的账期组合
        for payment_terms in range(30, 91, 15):  # 30到90天，步长15天
            for collection_days in range(30, 61, 15):  # 30到60天
                prediction = self.predict(
                    inventory_plan=inventory_plan,
                    unit_costs=unit_costs,
                    sales_forecast=sales_forecast,
                    selling_prices=selling_prices,
                    payment_terms_days=payment_terms,
                    collection_days=collection_days
                )

                # 如果营运资本需求更低且满足目标
                if (prediction.working_capital_needs < best_terms['working_capital_needs']
                    and prediction.working_capital_needs <= target_working_capital):
                    best_terms = {
                        'payment_terms_days': payment_terms,
                        'collection_days': collection_days,
                        'working_capital_needs': prediction.working_capital_needs
                    }

        return best_terms
