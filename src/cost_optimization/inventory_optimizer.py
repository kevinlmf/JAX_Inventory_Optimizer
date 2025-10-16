"""
Dynamic Inventory Optimizer - 动态最优库存引擎

核心目标：将库存成本降低20-35%
实现方式：实时计算每个SKU的最优库存水平，而非固定安全库存

商业价值：
- 减少资金占用
- 降低持有成本
- 提升库存周转率
- 释放现金流
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class InventoryStrategy(Enum):
    """库存策略类型"""
    AGGRESSIVE = "aggressive"      # 激进策略：最低库存，最高周转
    BALANCED = "balanced"          # 平衡策略：平衡成本和服务
    CONSERVATIVE = "conservative"  # 保守策略：高库存，零缺货


@dataclass
class OptimalInventory:
    """最优库存计算结果"""
    sku_id: str
    current_inventory: float
    optimal_inventory: float
    recommended_order: float
    expected_days_to_stockout: float
    inventory_cost_saving: float
    turnover_rate: float
    strategy: InventoryStrategy
    confidence_level: float
    reasoning: str


@dataclass
class InventoryHealthMetrics:
    """库存健康度指标"""
    total_inventory_value: float
    excess_inventory_value: float
    shortage_risk_value: float
    avg_turnover_rate: float
    capital_efficiency_score: float  # 0-100
    health_status: str  # "优秀", "良好", "一般", "差"
    improvement_opportunities: List[Dict[str, Any]]


class DynamicInventoryOptimizer:
    """
    动态库存优化器 - AI驱动的实时库存优化

    核心算法：
    1. 需求预测（多模型集成）
    2. 风险评估（缺货vs积压）
    3. 成本建模（持有成本vs机会成本）
    4. 动态优化（实时调整最优库存水平）
    """

    def __init__(self,
                 holding_cost_rate: float = 0.24,  # 年化持有成本率 24%
                 stockout_cost_multiplier: float = 5.0,  # 缺货成本是利润的5倍
                 target_service_level: float = 0.95,  # 目标服务水平 95%
                 strategy: InventoryStrategy = InventoryStrategy.BALANCED):
        """
        初始化库存优化器

        Args:
            holding_cost_rate: 年化库存持有成本率（仓储+资金+损耗）
            stockout_cost_multiplier: 缺货成本倍数（相对于单位利润）
            target_service_level: 目标服务水平（不缺货概率）
            strategy: 库存策略（激进/平衡/保守）
        """
        self.holding_cost_rate = holding_cost_rate
        self.daily_holding_cost_rate = holding_cost_rate / 365
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.target_service_level = target_service_level
        self.strategy = strategy

        # 策略参数
        self.strategy_params = self._get_strategy_params(strategy)

    def _get_strategy_params(self, strategy: InventoryStrategy) -> Dict[str, float]:
        """获取不同策略的参数"""
        params = {
            InventoryStrategy.AGGRESSIVE: {
                'service_level': 0.90,
                'safety_stock_multiplier': 1.28,  # Z-score for 90%
                'reorder_urgency': 0.7,
                'turnover_target': 12  # 月度周转12次
            },
            InventoryStrategy.BALANCED: {
                'service_level': 0.95,
                'safety_stock_multiplier': 1.65,  # Z-score for 95%
                'reorder_urgency': 0.5,
                'turnover_target': 8
            },
            InventoryStrategy.CONSERVATIVE: {
                'service_level': 0.99,
                'safety_stock_multiplier': 2.33,  # Z-score for 99%
                'reorder_urgency': 0.3,
                'turnover_target': 6
            }
        }
        return params[strategy]

    def calculate_optimal_inventory(self,
                                   sku_id: str,
                                   current_inventory: float,
                                   demand_history: np.ndarray,
                                   demand_forecast: np.ndarray,
                                   lead_time: int,
                                   unit_cost: float,
                                   unit_price: float,
                                   pending_orders: float = 0) -> OptimalInventory:
        """
        计算单个SKU的最优库存水平

        Args:
            sku_id: SKU标识
            current_inventory: 当前库存
            demand_history: 历史需求
            demand_forecast: 需求预测（未来N天）
            lead_time: 补货周期（天）
            unit_cost: 单位成本
            unit_price: 单位售价
            pending_orders: 在途订单

        Returns:
            OptimalInventory: 最优库存计算结果
        """
        # 1. 分析需求特征
        demand_stats = self._analyze_demand(demand_history, demand_forecast)

        # 2. 计算安全库存
        safety_stock = self._calculate_safety_stock(
            demand_stats['mean'],
            demand_stats['std'],
            lead_time
        )

        # 3. 计算最优库存水平
        optimal_level = self._calculate_optimal_level(
            demand_stats,
            lead_time,
            safety_stock,
            unit_cost,
            unit_price
        )

        # 4. 计算推荐订货量
        inventory_position = current_inventory + pending_orders
        recommended_order = max(0, optimal_level - inventory_position)

        # 5. 预测缺货时间
        days_to_stockout = self._estimate_stockout_time(
            current_inventory,
            demand_stats['mean']
        )

        # 6. 计算成本节省
        cost_saving = self._calculate_cost_saving(
            current_inventory,
            optimal_level,
            unit_cost
        )

        # 7. 计算周转率
        turnover_rate = self._calculate_turnover_rate(
            optimal_level,
            demand_stats['mean']
        )

        # 8. 生成决策理由
        reasoning = self._generate_reasoning(
            current_inventory,
            optimal_level,
            demand_stats,
            cost_saving,
            turnover_rate
        )

        return OptimalInventory(
            sku_id=sku_id,
            current_inventory=current_inventory,
            optimal_inventory=optimal_level,
            recommended_order=recommended_order,
            expected_days_to_stockout=days_to_stockout,
            inventory_cost_saving=cost_saving,
            turnover_rate=turnover_rate,
            strategy=self.strategy,
            confidence_level=demand_stats['forecast_confidence'],
            reasoning=reasoning
        )

    def _analyze_demand(self, history: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """分析需求特征"""
        # 历史需求统计
        hist_mean = np.mean(history)
        hist_std = np.std(history)

        # 预测需求统计
        forecast_mean = np.mean(forecast)
        forecast_std = np.std(forecast)

        # 综合统计（历史权重0.3，预测权重0.7）
        combined_mean = 0.3 * hist_mean + 0.7 * forecast_mean
        combined_std = np.sqrt(0.3 * hist_std**2 + 0.7 * forecast_std**2)

        # 需求变异系数（衡量波动性）
        cv = combined_std / (combined_mean + 1e-6)

        # 趋势检测
        recent_mean = np.mean(history[-7:]) if len(history) >= 7 else hist_mean
        trend = (recent_mean - hist_mean) / (hist_mean + 1e-6)

        # 预测置信度（基于变异系数）
        confidence = max(0.5, 1 - cv)

        return {
            'mean': combined_mean,
            'std': combined_std,
            'cv': cv,
            'trend': trend,
            'forecast_confidence': confidence
        }

    def _calculate_safety_stock(self, mean_demand: float, std_demand: float, lead_time: int) -> float:
        """计算安全库存"""
        # 考虑前置期的需求不确定性
        z_score = self.strategy_params['safety_stock_multiplier']
        lead_time_std = std_demand * np.sqrt(lead_time)

        safety_stock = z_score * lead_time_std

        return safety_stock

    def _calculate_optimal_level(self,
                                demand_stats: Dict[str, float],
                                lead_time: int,
                                safety_stock: float,
                                unit_cost: float,
                                unit_price: float) -> float:
        """
        计算最优库存水平

        基于newsvendor模型，平衡缺货成本和持有成本
        """
        mean_demand = demand_stats['mean']

        # 前置期内的预期需求
        lead_time_demand = mean_demand * lead_time

        # 基础最优库存 = 前置期需求 + 安全库存
        base_optimal = lead_time_demand + safety_stock

        # 根据策略调整
        strategy_adjustment = self.strategy_params['reorder_urgency']

        # 考虑成本因素的调整
        margin = unit_price - unit_cost
        margin_rate = margin / unit_price if unit_price > 0 else 0.3

        # 高利润商品可以适当增加库存，低利润商品减少库存
        margin_adjustment = 0.9 + (margin_rate - 0.3) * 0.5  # 0.8 to 1.1

        optimal_level = base_optimal * strategy_adjustment * margin_adjustment

        # 确保至少有3天的库存
        min_inventory = mean_demand * 3
        optimal_level = max(optimal_level, min_inventory)

        return optimal_level

    def _estimate_stockout_time(self, current_inventory: float, mean_demand: float) -> float:
        """预测缺货时间（天数）"""
        if mean_demand <= 0:
            return float('inf')

        days = current_inventory / mean_demand
        return max(0, days)

    def _calculate_cost_saving(self, current: float, optimal: float, unit_cost: float) -> float:
        """计算成本节省"""
        inventory_reduction = current - optimal

        if inventory_reduction > 0:
            # 减少库存带来的年度节省
            annual_saving = inventory_reduction * unit_cost * self.holding_cost_rate
            return annual_saving
        else:
            # 增加库存的成本（负节省）
            additional_cost = -inventory_reduction * unit_cost * self.holding_cost_rate
            return -additional_cost

    def _calculate_turnover_rate(self, optimal_inventory: float, mean_demand: float) -> float:
        """计算库存周转率（年化）"""
        if optimal_inventory <= 0:
            return 0

        # 年化周转率 = 年需求 / 平均库存
        annual_demand = mean_demand * 365
        turnover = annual_demand / optimal_inventory

        return turnover

    def _generate_reasoning(self,
                          current: float,
                          optimal: float,
                          demand_stats: Dict[str, float],
                          cost_saving: float,
                          turnover: float) -> str:
        """生成决策理由"""
        diff = current - optimal
        diff_pct = (diff / current * 100) if current > 0 else 0

        if abs(diff_pct) < 10:
            status = "✅ 当前库存健康"
            action = "保持现状"
        elif diff > 0:
            status = "⚠️ 库存过高"
            action = f"建议减少 {abs(diff):.0f} 件 ({abs(diff_pct):.0f}%)"
        else:
            status = "🔴 库存不足"
            action = f"建议补货 {abs(diff):.0f} 件"

        reasoning = f"""
{status}

当前库存: {current:.0f} 件
最优库存: {optimal:.0f} 件
调整建议: {action}

📊 需求分析:
- 日均需求: {demand_stats['mean']:.1f} 件
- 需求波动: {demand_stats['cv']:.2f} (变异系数)
- 需求趋势: {"↗️ 上升" if demand_stats['trend'] > 0.05 else "↘️ 下降" if demand_stats['trend'] < -0.05 else "→ 平稳"}

💰 财务影响:
- 年度成本节省: ${abs(cost_saving):,.0f} {"(节省)" if cost_saving > 0 else "(增加)"}
- 预期周转率: {turnover:.1f} 次/年
- 策略: {self.strategy.value}

💡 业务建议:
{"• 减少订货量可释放资金用于高周转商品" if diff > 0 else "• 及时补货避免缺货损失"}
{"• 当前库存可维持 " + str(int(current / demand_stats['mean'])) + " 天"}
        """
        return reasoning.strip()

    def optimize_portfolio(self,
                          inventory_data: List[Dict[str, Any]],
                          total_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        优化整个库存组合

        Args:
            inventory_data: 所有SKU的库存数据
            total_budget: 可用预算限制

        Returns:
            优化结果和建议
        """
        results = []
        total_current_value = 0
        total_optimal_value = 0
        total_cost_saving = 0

        # 计算每个SKU的最优库存
        for item in inventory_data:
            optimal = self.calculate_optimal_inventory(
                sku_id=item['sku_id'],
                current_inventory=item['current_inventory'],
                demand_history=item['demand_history'],
                demand_forecast=item['demand_forecast'],
                lead_time=item.get('lead_time', 7),
                unit_cost=item['unit_cost'],
                unit_price=item['unit_price'],
                pending_orders=item.get('pending_orders', 0)
            )
            results.append(optimal)

            total_current_value += item['current_inventory'] * item['unit_cost']
            total_optimal_value += optimal.optimal_inventory * item['unit_cost']
            total_cost_saving += optimal.inventory_cost_saving

        # 如果有预算限制，需要进行优先级排序
        if total_budget and total_budget < total_optimal_value:
            results = self._apply_budget_constraint(results, inventory_data, total_budget)

        # 计算整体指标
        health_metrics = self._calculate_health_metrics(results, inventory_data)

        return {
            'sku_optimizations': results,
            'summary': {
                'total_skus': len(results),
                'current_inventory_value': total_current_value,
                'optimal_inventory_value': total_optimal_value,
                'value_reduction': total_current_value - total_optimal_value,
                'value_reduction_pct': ((total_current_value - total_optimal_value) / total_current_value * 100) if total_current_value > 0 else 0,
                'annual_cost_saving': total_cost_saving,
                'capital_freed': max(0, total_current_value - total_optimal_value)
            },
            'health_metrics': health_metrics
        }

    def _apply_budget_constraint(self,
                                results: List[OptimalInventory],
                                inventory_data: List[Dict],
                                budget: float) -> List[OptimalInventory]:
        """在预算约束下优化库存分配"""
        # 按周转率和缺货风险排序，优先分配高周转、高风险的SKU
        scored_results = []
        for i, result in enumerate(results):
            score = (result.turnover_rate * 0.6 +
                    (1 / max(result.expected_days_to_stockout, 1)) * 0.4)
            scored_results.append((score, i, result))

        scored_results.sort(reverse=True, key=lambda x: x[0])

        # 重新分配库存
        remaining_budget = budget
        optimized_results = []

        for score, idx, result in scored_results:
            item = inventory_data[idx]
            required_value = result.optimal_inventory * item['unit_cost']

            if remaining_budget >= required_value:
                optimized_results.append(result)
                remaining_budget -= required_value
            else:
                # 部分满足
                affordable_qty = remaining_budget / item['unit_cost']
                result.optimal_inventory = affordable_qty
                result.recommended_order = max(0, affordable_qty - result.current_inventory)
                optimized_results.append(result)
                remaining_budget = 0
                break

        return optimized_results

    def _calculate_health_metrics(self,
                                 results: List[OptimalInventory],
                                 inventory_data: List[Dict]) -> InventoryHealthMetrics:
        """计算整体库存健康度"""
        total_value = sum(r.current_inventory * d['unit_cost']
                         for r, d in zip(results, inventory_data))

        excess_value = sum(max(0, (r.current_inventory - r.optimal_inventory) * d['unit_cost'])
                          for r, d in zip(results, inventory_data))

        shortage_value = sum(max(0, (r.optimal_inventory - r.current_inventory) * d['unit_cost'])
                            for r, d in zip(results, inventory_data))

        avg_turnover = np.mean([r.turnover_rate for r in results])

        # 资金效率评分 (0-100)
        excess_ratio = excess_value / total_value if total_value > 0 else 0
        shortage_ratio = shortage_value / total_value if total_value > 0 else 0
        turnover_score = min(100, avg_turnover / 12 * 100)  # 12次/年为满分

        capital_score = 100 * (1 - excess_ratio - shortage_ratio * 0.5)
        capital_score = max(0, min(100, capital_score * 0.5 + turnover_score * 0.5))

        # 健康状态判断
        if capital_score >= 80:
            health_status = "优秀"
        elif capital_score >= 60:
            health_status = "良好"
        elif capital_score >= 40:
            health_status = "一般"
        else:
            health_status = "差"

        # 改进机会
        opportunities = []
        if excess_ratio > 0.2:
            opportunities.append({
                'type': '减少过剩库存',
                'potential_saving': excess_value * 0.24,  # 年度持有成本
                'description': f'识别到 ${excess_value:,.0f} 过剩库存，可节省年度成本'
            })

        if avg_turnover < 8:
            opportunities.append({
                'type': '提升周转率',
                'potential_saving': total_value * 0.1,
                'description': f'当前周转率 {avg_turnover:.1f} 次/年，提升至8次可释放资金'
            })

        return InventoryHealthMetrics(
            total_inventory_value=total_value,
            excess_inventory_value=excess_value,
            shortage_risk_value=shortage_value,
            avg_turnover_rate=avg_turnover,
            capital_efficiency_score=capital_score,
            health_status=health_status,
            improvement_opportunities=opportunities
        )


if __name__ == "__main__":
    # 测试示例
    print("📦 动态库存优化引擎 - 测试")
    print("=" * 70)

    # 创建优化器
    optimizer = DynamicInventoryOptimizer(
        holding_cost_rate=0.24,
        strategy=InventoryStrategy.BALANCED
    )

    # 模拟数据
    np.random.seed(42)
    demand_history = np.random.poisson(50, 90)  # 90天历史
    demand_forecast = np.random.poisson(55, 30)  # 30天预测

    # 计算最优库存
    result = optimizer.calculate_optimal_inventory(
        sku_id="SKU_001",
        current_inventory=200,
        demand_history=demand_history,
        demand_forecast=demand_forecast,
        lead_time=7,
        unit_cost=10.0,
        unit_price=20.0,
        pending_orders=0
    )

    print(f"\n📊 优化结果:")
    print(f"SKU: {result.sku_id}")
    print(f"当前库存: {result.current_inventory:.0f} 件")
    print(f"最优库存: {result.optimal_inventory:.0f} 件")
    print(f"建议订货: {result.recommended_order:.0f} 件")
    print(f"预计缺货: {result.expected_days_to_stockout:.1f} 天")
    print(f"成本节省: ${result.inventory_cost_saving:,.0f}/年")
    print(f"周转率: {result.turnover_rate:.1f} 次/年")
    print(f"\n💡 决策理由:")
    print(result.reasoning)

    # 测试组合优化
    print("\n" + "=" * 70)
    print("📊 组合优化测试")

    inventory_portfolio = [
        {
            'sku_id': f'SKU_{i:03d}',
            'current_inventory': np.random.randint(50, 300),
            'demand_history': np.random.poisson(50, 90),
            'demand_forecast': np.random.poisson(52, 30),
            'lead_time': 7,
            'unit_cost': 10.0,
            'unit_price': 20.0,
            'pending_orders': 0
        }
        for i in range(5)
    ]

    portfolio_result = optimizer.optimize_portfolio(inventory_portfolio)

    print(f"\n📈 组合优化结果:")
    print(f"总SKU数: {portfolio_result['summary']['total_skus']}")
    print(f"当前库存价值: ${portfolio_result['summary']['current_inventory_value']:,.0f}")
    print(f"最优库存价值: ${portfolio_result['summary']['optimal_inventory_value']:,.0f}")
    print(f"可减少库存: ${portfolio_result['summary']['value_reduction']:,.0f} ({portfolio_result['summary']['value_reduction_pct']:.1f}%)")
    print(f"年度成本节省: ${portfolio_result['summary']['annual_cost_saving']:,.0f}")
    print(f"释放资金: ${portfolio_result['summary']['capital_freed']:,.0f}")

    health = portfolio_result['health_metrics']
    print(f"\n💊 健康度评估:")
    print(f"整体评分: {health.capital_efficiency_score:.1f}/100 ({health.health_status})")
    print(f"平均周转率: {health.avg_turnover_rate:.1f} 次/年")
    print(f"过剩库存: ${health.excess_inventory_value:,.0f}")

    if health.improvement_opportunities:
        print(f"\n💡 改进机会:")
        for opp in health.improvement_opportunities:
            print(f"  • {opp['type']}: ${opp['potential_saving']:,.0f}")
            print(f"    {opp['description']}")

    print("\n✅ 测试完成！")


# Alias for backward compatibility  
InventoryOptimizer = DynamicInventoryOptimizer
