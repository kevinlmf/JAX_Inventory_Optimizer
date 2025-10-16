"""
JIT (Just-In-Time) Ordering Optimizer
优化订货时机，减少库存持有成本
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class JITOptimizationResult:
    """JIT优化结果"""
    optimal_order_point: float
    optimal_order_quantity: float
    expected_cost_savings: float
    service_level: float
    risk_score: float
    recommendations: Dict[str, Any]


class JITOrderingOptimizer:
    """
    JIT订货优化器

    核心功能：
    1. 动态计算最佳订货点
    2. 优化订货批量
    3. 平衡库存成本与服务水平
    """

    def __init__(
        self,
        target_service_level: float = 0.95,
        holding_cost_rate: float = 0.25,
        ordering_cost: float = 100.0,
        stockout_cost: float = None
    ):
        """
        初始化JIT优化器

        Args:
            target_service_level: 目标服务水平 (0-1)
            holding_cost_rate: 库存持有成本率 (年化)
            ordering_cost: 单次订货成本
            stockout_cost: 缺货成本 (optional, for compatibility)
        """
        self.target_service_level = target_service_level
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost if stockout_cost is not None else 50.0

    def optimize(
        self,
        demand_forecast: jnp.ndarray,
        demand_std: float,
        lead_time: int,
        unit_cost: float,
        current_inventory: float
    ) -> JITOptimizationResult:
        """
        优化JIT订货策略

        Args:
            demand_forecast: 需求预测序列
            demand_std: 需求标准差
            lead_time: 订货提前期（天）
            unit_cost: 单位成本
            current_inventory: 当前库存

        Returns:
            JITOptimizationResult: 优化结果
        """
        # 计算提前期需求
        lead_time_demand = self._calculate_lead_time_demand(
            demand_forecast, lead_time
        )

        # 计算安全库存
        safety_stock = self._calculate_safety_stock(
            demand_std, lead_time, self.target_service_level
        )

        # 计算最优订货点
        optimal_order_point = lead_time_demand + safety_stock

        # 计算最优订货批量 (EOQ公式)
        annual_demand = jnp.sum(demand_forecast)
        optimal_order_quantity = self._calculate_eoq(
            annual_demand, self.ordering_cost, unit_cost, self.holding_cost_rate
        )

        # 计算成本节约
        cost_savings = self._estimate_cost_savings(
            current_inventory,
            optimal_order_point,
            optimal_order_quantity,
            unit_cost,
            self.holding_cost_rate
        )

        # 评估风险
        risk_score = self._assess_risk(
            current_inventory,
            optimal_order_point,
            demand_std,
            lead_time
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            current_inventory,
            optimal_order_point,
            optimal_order_quantity,
            risk_score
        )

        return JITOptimizationResult(
            optimal_order_point=float(optimal_order_point),
            optimal_order_quantity=float(optimal_order_quantity),
            expected_cost_savings=float(cost_savings),
            service_level=self.target_service_level,
            risk_score=float(risk_score),
            recommendations=recommendations
        )

    def _calculate_lead_time_demand(
        self,
        demand_forecast: jnp.ndarray,
        lead_time: int
    ) -> float:
        """计算提前期需求"""
        # 取未来lead_time天的平均日需求
        daily_demand = jnp.mean(demand_forecast[:min(len(demand_forecast), 30)])
        return daily_demand * lead_time

    def _calculate_safety_stock(
        self,
        demand_std: float,
        lead_time: int,
        service_level: float
    ) -> float:
        """
        计算安全库存
        Safety Stock = Z * σ * sqrt(L)
        """
        # Z-score for service level
        z_score = self._get_z_score(service_level)
        safety_stock = z_score * demand_std * jnp.sqrt(lead_time)
        return safety_stock

    def _get_z_score(self, service_level: float) -> float:
        """获取服务水平对应的Z分数"""
        # 常见服务水平的Z分数映射
        if service_level >= 0.99:
            return 2.33
        elif service_level >= 0.95:
            return 1.65
        elif service_level >= 0.90:
            return 1.28
        else:
            return 1.0

    def _calculate_eoq(
        self,
        annual_demand: float,
        ordering_cost: float,
        unit_cost: float,
        holding_cost_rate: float
    ) -> float:
        """
        计算经济订货批量 (EOQ)
        EOQ = sqrt(2 * D * S / (H * C))
        """
        holding_cost = holding_cost_rate * unit_cost
        eoq = jnp.sqrt(
            2 * annual_demand * ordering_cost / holding_cost
        )
        return eoq

    def _estimate_cost_savings(
        self,
        current_inventory: float,
        optimal_order_point: float,
        optimal_order_quantity: float,
        unit_cost: float,
        holding_cost_rate: float
    ) -> float:
        """估算成本节约"""
        # 当前库存持有成本
        current_holding_cost = current_inventory * unit_cost * holding_cost_rate

        # 优化后的平均库存（订货点 + 订货批量/2）
        optimal_avg_inventory = optimal_order_point + optimal_order_quantity / 2
        optimal_holding_cost = optimal_avg_inventory * unit_cost * holding_cost_rate

        # 成本节约（年化）
        cost_savings = jnp.maximum(0, current_holding_cost - optimal_holding_cost)
        return cost_savings

    def _assess_risk(
        self,
        current_inventory: float,
        optimal_order_point: float,
        demand_std: float,
        lead_time: int
    ) -> float:
        """
        评估风险水平
        返回0-1之间的风险分数，1表示高风险
        """
        # 当前库存与订货点的差距
        inventory_gap = optimal_order_point - current_inventory

        # 需求波动性
        volatility = demand_std * jnp.sqrt(lead_time)

        # 风险分数
        if inventory_gap <= 0:
            risk = 0.1  # 库存充足，低风险
        else:
            # 基于缺口占波动性的比例计算风险
            risk = jnp.minimum(1.0, inventory_gap / (2 * volatility))

        return float(risk)

    def _generate_recommendations(
        self,
        current_inventory: float,
        optimal_order_point: float,
        optimal_order_quantity: float,
        risk_score: float
    ) -> Dict[str, Any]:
        """生成优化建议"""
        recommendations = {
            'action': 'HOLD',
            'urgency': 'LOW',
            'reason': '',
            'suggested_order_quantity': 0.0
        }

        if current_inventory < optimal_order_point:
            recommendations['action'] = 'ORDER'
            recommendations['suggested_order_quantity'] = float(optimal_order_quantity)

            if risk_score > 0.7:
                recommendations['urgency'] = 'HIGH'
                recommendations['reason'] = '库存即将不足，建议立即订货'
            elif risk_score > 0.4:
                recommendations['urgency'] = 'MEDIUM'
                recommendations['reason'] = '库存偏低，建议尽快订货'
            else:
                recommendations['urgency'] = 'LOW'
                recommendations['reason'] = '库存接近订货点，可计划订货'
        else:
            recommendations['reason'] = '当前库存充足，无需订货'

        return recommendations

    def batch_optimize(
        self,
        products_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, JITOptimizationResult]:
        """
        批量优化多个产品的JIT策略

        Args:
            products_data: 产品数据字典，key为产品ID

        Returns:
            Dict[str, JITOptimizationResult]: 每个产品的优化结果
        """
        results = {}

        for product_id, data in products_data.items():
            result = self.optimize(
                demand_forecast=jnp.array(data['demand_forecast']),
                demand_std=data['demand_std'],
                lead_time=data['lead_time'],
                unit_cost=data['unit_cost'],
                current_inventory=data['current_inventory']
            )
            results[product_id] = result

        return results

    def optimize_order_quantity(
        self,
        inventory_levels: jnp.ndarray,
        demand_forecasts: jnp.ndarray,
        lead_times: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Vectorized order quantity optimization for multiple products

        Args:
            inventory_levels: Current inventory levels for each product
            demand_forecasts: Demand forecasts for each product
            lead_times: Lead times for each product

        Returns:
            Optimal order quantities for each product
        """
        # Calculate lead time demand
        lead_time_demand = demand_forecasts * lead_times

        # Calculate safety stock (simplified Z-score approach)
        z_score = self._get_z_score(self.target_service_level)
        demand_std = demand_forecasts * 0.2  # Assume 20% CV
        safety_stock = z_score * demand_std * jnp.sqrt(lead_times)

        # Calculate reorder point
        reorder_point = lead_time_demand + safety_stock

        # Determine if order is needed
        should_order = inventory_levels <= reorder_point

        # Calculate order quantities (EOQ-based)
        annual_demand = demand_forecasts * 365
        unit_cost = 10.0  # Default unit cost
        holding_cost = self.holding_cost_rate * unit_cost

        # EOQ formula
        eoq = jnp.sqrt(2 * annual_demand * self.ordering_cost / holding_cost)

        # Return order quantity only if below reorder point
        order_quantities = jnp.where(should_order, eoq, 0.0)

        return order_quantities


# Alias for backward compatibility
JITOptimizer = JITOrderingOptimizer
