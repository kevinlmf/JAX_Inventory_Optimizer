"""
Auto Decision Engine
自动决策引擎 - 减少沟通成本，提高决策效率
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DecisionType(Enum):
    """决策类型"""
    REPLENISHMENT = "replenishment"  # 补货决策
    CLEARANCE = "clearance"  # 清仓决策
    PRICING = "pricing"  # 定价决策
    ALLOCATION = "allocation"  # 分配决策


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"  # 高置信度，自动执行
    MEDIUM = "medium"  # 中等置信度，建议执行
    LOW = "low"  # 低置信度，需人工审核


@dataclass
class Decision:
    """决策结果"""
    decision_id: str
    decision_type: DecisionType
    confidence_level: ConfidenceLevel
    confidence_score: float
    action: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    risks: List[str]
    auto_executable: bool
    explanation: str


class AutoDecisionEngine:
    """
    自动决策引擎

    核心价值：
    1. 减少沟通成本 - AI自动决策，无需层层审批
    2. 提高响应速度 - 实时决策，抓住市场机会
    3. 降低人工成本 - 减少重复性决策工作
    4. 提升决策质量 - 基于数据，减少主观偏差

    ROI体现：
    - 决策时间从天级降至秒级
    - 减少决策会议和沟通成本
    - 提高决策准确率
    """

    def __init__(
        self,
        auto_execute_threshold: float = 0.85,
        risk_tolerance: str = "moderate"
    ):
        """
        初始化自动决策引擎

        Args:
            auto_execute_threshold: 自动执行的置信度阈值
            risk_tolerance: 风险容忍度 (conservative/moderate/aggressive)
        """
        self.auto_execute_threshold = auto_execute_threshold
        self.risk_tolerance = risk_tolerance

        # 风险容忍度映射
        self.risk_params = {
            'conservative': {'threshold': 0.90, 'max_order_amount': 50000},
            'moderate': {'threshold': 0.85, 'max_order_amount': 100000},
            'aggressive': {'threshold': 0.75, 'max_order_amount': 200000}
        }[risk_tolerance]

    def make_replenishment_decision(
        self,
        product_id: str,
        current_inventory: float,
        demand_forecast: jnp.ndarray,
        demand_std: float,
        lead_time: int,
        unit_cost: float,
        holding_cost_rate: float,
        service_level: float = 0.95
    ) -> Decision:
        """
        补货决策

        Args:
            product_id: 产品ID
            current_inventory: 当前库存
            demand_forecast: 需求预测
            demand_std: 需求标准差
            lead_time: 提前期
            unit_cost: 单位成本
            holding_cost_rate: 持有成本率
            service_level: 目标服务水平

        Returns:
            Decision: 决策结果
        """
        # 计算提前期需求
        daily_demand = float(jnp.mean(demand_forecast[:30]))
        lead_time_demand = daily_demand * lead_time

        # 计算安全库存
        z_score = 1.65 if service_level >= 0.95 else 1.28
        safety_stock = z_score * demand_std * jnp.sqrt(lead_time)

        # 计算再订货点
        reorder_point = lead_time_demand + safety_stock

        # 计算EOQ
        annual_demand = daily_demand * 365
        eoq = jnp.sqrt(2 * annual_demand * 100 / (holding_cost_rate * unit_cost))

        # 决策逻辑
        if current_inventory < reorder_point:
            order_quantity = float(eoq)
            action = "ORDER"

            # 评估置信度
            confidence_score = self._calculate_confidence(
                current_inventory,
                reorder_point,
                demand_std,
                demand_forecast
            )

            # 评估影响
            expected_impact = {
                'inventory_increase': order_quantity,
                'cost': order_quantity * unit_cost,
                'stockout_risk_reduction': 0.8,
                'holding_cost_increase': order_quantity * unit_cost * holding_cost_rate / 2
            }

            # 评估风险
            risks = []
            if demand_std > daily_demand * 0.3:
                risks.append("需求波动较大，可能导致库存过剩")
            if order_quantity * unit_cost > self.risk_params['max_order_amount']:
                risks.append("订货金额超过风险阈值")
                confidence_score *= 0.8

            # 决定是否自动执行
            auto_executable = (
                confidence_score >= self.risk_params['threshold'] and
                order_quantity * unit_cost <= self.risk_params['max_order_amount'] and
                len(risks) == 0
            )

            explanation = (
                f"当前库存{current_inventory:.0f}低于再订货点{reorder_point:.0f}，"
                f"建议订货{order_quantity:.0f}单位，预计成本{order_quantity * unit_cost:.0f}元"
            )

        else:
            action = "HOLD"
            order_quantity = 0
            confidence_score = 0.95
            expected_impact = {'inventory_change': 0}
            risks = []
            auto_executable = True
            explanation = f"当前库存{current_inventory:.0f}充足，无需订货"

        return Decision(
            decision_id=f"REP_{product_id}_{jnp.random.randint(0, 10000)}",
            decision_type=DecisionType.REPLENISHMENT,
            confidence_level=self._get_confidence_level(confidence_score),
            confidence_score=float(confidence_score),
            action=action,
            parameters={
                'product_id': product_id,
                'order_quantity': float(order_quantity),
                'reorder_point': float(reorder_point),
                'safety_stock': float(safety_stock)
            },
            expected_impact=expected_impact,
            risks=risks,
            auto_executable=auto_executable,
            explanation=explanation
        )

    def make_clearance_decision(
        self,
        product_id: str,
        current_inventory: float,
        days_no_sale: int,
        unit_cost: float,
        current_price: float,
        storage_cost_per_day: float
    ) -> Decision:
        """
        清仓决策

        判断是否需要清仓及清仓价格
        """
        # 计算持有成本
        holding_cost = days_no_sale * storage_cost_per_day

        # 决策逻辑
        if days_no_sale > 90:  # 90天无销售
            # 建议清仓
            # 清仓价格：成本价 - 持有成本
            clearance_price = max(unit_cost * 0.5, unit_cost - holding_cost)
            discount_rate = (current_price - clearance_price) / current_price

            confidence_score = 0.9 if days_no_sale > 180 else 0.75

            expected_impact = {
                'inventory_reduction': current_inventory,
                'revenue': current_inventory * clearance_price,
                'loss': current_inventory * (unit_cost - clearance_price),
                'space_freed': current_inventory,
                'holding_cost_saved': current_inventory * storage_cost_per_day * 30
            }

            risks = []
            if discount_rate > 0.5:
                risks.append("折扣力度较大，可能影响品牌形象")
                confidence_score *= 0.9

            auto_executable = (
                confidence_score >= self.risk_params['threshold'] and
                clearance_price >= unit_cost * 0.3  # 不低于成本的30%
            )

            explanation = (
                f"产品{days_no_sale}天无销售，建议以{clearance_price:.2f}元清仓"
                f"（折扣{discount_rate*100:.0f}%），回收{expected_impact['revenue']:.0f}元"
            )

            action = "CLEARANCE"
            parameters = {
                'product_id': product_id,
                'clearance_price': float(clearance_price),
                'discount_rate': float(discount_rate),
                'quantity': float(current_inventory)
            }

        else:
            action = "HOLD"
            confidence_score = 0.95
            expected_impact = {'inventory_change': 0}
            risks = []
            auto_executable = True
            explanation = f"产品{days_no_sale}天无销售，暂不需要清仓"
            parameters = {'product_id': product_id}

        return Decision(
            decision_id=f"CLR_{product_id}_{jnp.random.randint(0, 10000)}",
            decision_type=DecisionType.CLEARANCE,
            confidence_level=self._get_confidence_level(confidence_score),
            confidence_score=float(confidence_score),
            action=action,
            parameters=parameters,
            expected_impact=expected_impact,
            risks=risks,
            auto_executable=auto_executable,
            explanation=explanation
        )

    def make_pricing_decision(
        self,
        product_id: str,
        current_price: float,
        unit_cost: float,
        current_inventory: float,
        recent_sales: jnp.ndarray,
        price_elasticity: float = -1.5
    ) -> Decision:
        """
        动态定价决策

        基于库存水平和销售趋势调整价格
        """
        # 分析销售趋势
        avg_daily_sales = float(jnp.mean(recent_sales))
        sales_trend = float(jnp.mean(recent_sales[-7:]) - jnp.mean(recent_sales[:7]))

        # 计算库存周转率
        if avg_daily_sales > 0:
            days_of_inventory = current_inventory / avg_daily_sales
        else:
            days_of_inventory = 999

        # 定价策略
        if days_of_inventory > 60:  # 库存过高
            # 降价促销
            price_adjustment = -0.1  # 降价10%
            reason = "库存过高，降价促销"
        elif days_of_inventory < 15 and sales_trend > 0:  # 库存紧张且需求旺盛
            # 提价
            price_adjustment = 0.05  # 涨价5%
            reason = "需求旺盛，库存紧张，适当提价"
        else:
            # 保持价格
            price_adjustment = 0.0
            reason = "库存和销售平衡，维持现价"

        new_price = current_price * (1 + price_adjustment)

        # 预估影响
        expected_sales_change = price_adjustment * price_elasticity  # 价格弹性
        expected_revenue_change = (1 + price_adjustment) * (1 + expected_sales_change) - 1

        confidence_score = 0.8 if abs(price_adjustment) > 0 else 0.95

        expected_impact = {
            'price_change': float(new_price - current_price),
            'price_change_pct': float(price_adjustment),
            'expected_sales_change_pct': float(expected_sales_change),
            'expected_revenue_change_pct': float(expected_revenue_change)
        }

        risks = []
        if price_adjustment < -0.15:
            risks.append("降价幅度较大，可能影响利润")
        if price_adjustment > 0.1:
            risks.append("涨价幅度较大，可能影响销量")

        auto_executable = (
            confidence_score >= self.risk_params['threshold'] and
            abs(price_adjustment) <= 0.15  # 价格调整不超过15%
        )

        return Decision(
            decision_id=f"PRC_{product_id}_{jnp.random.randint(0, 10000)}",
            decision_type=DecisionType.PRICING,
            confidence_level=self._get_confidence_level(confidence_score),
            confidence_score=float(confidence_score),
            action="ADJUST_PRICE" if price_adjustment != 0 else "HOLD",
            parameters={
                'product_id': product_id,
                'current_price': float(current_price),
                'new_price': float(new_price),
                'adjustment_pct': float(price_adjustment)
            },
            expected_impact=expected_impact,
            risks=risks,
            auto_executable=auto_executable,
            explanation=f"{reason}，建议价格调整至{new_price:.2f}元"
        )

    def batch_decisions(
        self,
        products_data: Dict[str, Dict[str, Any]],
        decision_types: List[DecisionType]
    ) -> Dict[str, List[Decision]]:
        """
        批量决策

        Args:
            products_data: 产品数据
            decision_types: 需要做的决策类型

        Returns:
            各类型的决策列表
        """
        results = {dt.value: [] for dt in decision_types}

        for product_id, data in products_data.items():
            if DecisionType.REPLENISHMENT in decision_types:
                decision = self.make_replenishment_decision(
                    product_id=product_id,
                    current_inventory=data['current_inventory'],
                    demand_forecast=jnp.array(data['demand_forecast']),
                    demand_std=data['demand_std'],
                    lead_time=data['lead_time'],
                    unit_cost=data['unit_cost'],
                    holding_cost_rate=data.get('holding_cost_rate', 0.25)
                )
                results[DecisionType.REPLENISHMENT.value].append(decision)

            if DecisionType.CLEARANCE in decision_types:
                decision = self.make_clearance_decision(
                    product_id=product_id,
                    current_inventory=data['current_inventory'],
                    days_no_sale=data.get('days_no_sale', 0),
                    unit_cost=data['unit_cost'],
                    current_price=data.get('current_price', data['unit_cost'] * 1.5),
                    storage_cost_per_day=data.get('storage_cost_per_day', 1.0)
                )
                results[DecisionType.CLEARANCE.value].append(decision)

        return results

    def _calculate_confidence(
        self,
        current_inventory: float,
        reorder_point: float,
        demand_std: float,
        demand_forecast: jnp.ndarray
    ) -> float:
        """计算决策置信度"""
        # 基础置信度
        base_confidence = 0.8

        # 根据库存缺口调整
        inventory_gap = reorder_point - current_inventory
        if inventory_gap > demand_std * 2:
            confidence = base_confidence + 0.15  # 明显低于订货点
        elif inventory_gap > 0:
            confidence = base_confidence + 0.05
        else:
            confidence = base_confidence

        # 根据需求稳定性调整
        cv = demand_std / float(jnp.mean(demand_forecast) + 1e-6)  # 变异系数
        if cv < 0.2:
            confidence += 0.05  # 需求稳定
        elif cv > 0.5:
            confidence -= 0.1  # 需求波动大

        return min(1.0, max(0.0, confidence))

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """获取置信度等级"""
        if score >= 0.85:
            return ConfidenceLevel.HIGH
        elif score >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def get_auto_executable_decisions(
        self,
        decisions: List[Decision]
    ) -> List[Decision]:
        """筛选可自动执行的决策"""
        return [d for d in decisions if d.auto_executable]

    def get_decision_summary(
        self,
        decisions: List[Decision]
    ) -> Dict[str, Any]:
        """生成决策摘要"""
        total = len(decisions)
        auto_executable = len([d for d in decisions if d.auto_executable])

        by_type = {}
        for decision in decisions:
            dt = decision.decision_type.value
            if dt not in by_type:
                by_type[dt] = {'total': 0, 'auto': 0, 'high_confidence': 0}
            by_type[dt]['total'] += 1
            if decision.auto_executable:
                by_type[dt]['auto'] += 1
            if decision.confidence_level == ConfidenceLevel.HIGH:
                by_type[dt]['high_confidence'] += 1

        return {
            'total_decisions': total,
            'auto_executable': auto_executable,
            'auto_executable_rate': auto_executable / total if total > 0 else 0,
            'by_type': by_type,
            'communication_cost_saved': auto_executable * 100,  # 每个自动决策节省100元沟通成本
            'time_saved_hours': auto_executable * 0.5  # 每个自动决策节省0.5小时
        }
