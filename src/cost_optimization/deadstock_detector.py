"""
Deadstock Detector - 呆滞品智能识别系统

核心目标：识别并处理呆滞库存，释放被占用资金

商业价值：
- 每年回收$10,000-50,000被占用资金
- 减少资金机会成本
- 降低仓储成本
- 避免库存贬值损失
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class DeadstockRiskLevel(Enum):
    """呆滞风险等级"""
    CRITICAL = "critical"  # 高风险：立即处理
    HIGH = "high"          # 中高风险：30天内处理
    MEDIUM = "medium"      # 中风险：60天内处理
    LOW = "low"            # 低风险：监控即可
    HEALTHY = "healthy"    # 健康：无风险


@dataclass
class DeadstockAnalysis:
    """呆滞品分析结果"""
    sku_id: str
    risk_level: DeadstockRiskLevel
    risk_score: float  # 0-100
    current_inventory: float
    days_of_supply: float
    turnover_rate: float
    inventory_age_days: float
    tied_up_capital: float
    potential_loss: float
    recommended_action: str
    action_details: Dict[str, Any]
    urgency_days: int  # 建议处理时限（天）


class DeadstockDetector:
    """
    呆滞品检测器 - 多维度分析库存健康度

    检测维度：
    1. 周转率 - 库存周转速度
    2. 库存天数 - 当前库存可用天数
    3. 库龄 - 库存放置时间
    4. 需求趋势 - 需求是否下降
    5. 季节性 - 是否过季商品
    """

    def __init__(self,
                 turnover_threshold_critical: float = 1.0,  # 年周转<1次为高风险
                 turnover_threshold_high: float = 2.0,
                 turnover_threshold_medium: float = 4.0,
                 days_supply_threshold_critical: int = 90,
                 days_supply_threshold_high: int = 60,
                 days_supply_threshold_medium: int = 30,
                 inventory_age_threshold: int = 180):  # 库龄超过180天高风险
        """
        初始化呆滞品检测器

        Args:
            turnover_threshold_*: 不同风险等级的周转率阈值
            days_supply_threshold_*: 不同风险等级的供应天数阈值
            inventory_age_threshold: 库龄阈值（天）
        """
        self.turnover_critical = turnover_threshold_critical
        self.turnover_high = turnover_threshold_high
        self.turnover_medium = turnover_threshold_medium
        self.days_critical = days_supply_threshold_critical
        self.days_high = days_supply_threshold_high
        self.days_medium = days_supply_threshold_medium
        self.age_threshold = inventory_age_threshold

    def analyze_sku(self,
                   sku_id: str,
                   current_inventory: float,
                   demand_history: np.ndarray,
                   unit_cost: float,
                   unit_price: float,
                   inventory_age_days: int = 0,
                   last_sale_date: Optional[datetime] = None) -> DeadstockAnalysis:
        """
        分析单个SKU的呆滞风险

        Args:
            sku_id: SKU标识
            current_inventory: 当前库存量
            demand_history: 历史需求数据
            unit_cost: 单位成本
            unit_price: 单位售价
            inventory_age_days: 库龄（天）
            last_sale_date: 最后销售日期

        Returns:
            DeadstockAnalysis: 呆滞品分析结果
        """
        # 1. 计算周转率
        turnover_rate = self._calculate_turnover_rate(demand_history, current_inventory)

        # 2. 计算供应天数
        days_of_supply = self._calculate_days_of_supply(current_inventory, demand_history)

        # 3. 计算需求趋势
        demand_trend = self._analyze_demand_trend(demand_history)

        # 4. 计算风险评分
        risk_score = self._calculate_risk_score(
            turnover_rate,
            days_of_supply,
            inventory_age_days,
            demand_trend
        )

        # 5. 确定风险等级
        risk_level = self._determine_risk_level(risk_score, turnover_rate, days_of_supply)

        # 6. 计算财务影响
        tied_up_capital = current_inventory * unit_cost
        potential_loss = self._estimate_potential_loss(
            tied_up_capital,
            inventory_age_days,
            demand_trend,
            unit_cost,
            unit_price
        )

        # 7. 生成处理建议
        action = self._generate_action_plan(
            risk_level,
            current_inventory,
            turnover_rate,
            days_of_supply,
            unit_cost,
            unit_price,
            demand_trend
        )

        # 8. 确定处理时限
        urgency_days = self._calculate_urgency(risk_level, days_of_supply)

        return DeadstockAnalysis(
            sku_id=sku_id,
            risk_level=risk_level,
            risk_score=risk_score,
            current_inventory=current_inventory,
            days_of_supply=days_of_supply,
            turnover_rate=turnover_rate,
            inventory_age_days=inventory_age_days,
            tied_up_capital=tied_up_capital,
            potential_loss=potential_loss,
            recommended_action=action['action'],
            action_details=action['details'],
            urgency_days=urgency_days
        )

    def _calculate_turnover_rate(self, demand_history: np.ndarray, avg_inventory: float) -> float:
        """计算年化周转率"""
        if avg_inventory <= 0:
            return 0

        # 计算年化需求
        daily_demand = np.mean(demand_history) if len(demand_history) > 0 else 0
        annual_demand = daily_demand * 365

        # 周转率 = 年需求 / 平均库存
        turnover = annual_demand / avg_inventory
        return turnover

    def _calculate_days_of_supply(self, inventory: float, demand_history: np.ndarray) -> float:
        """计算供应天数"""
        daily_demand = np.mean(demand_history) if len(demand_history) > 0 else 0

        if daily_demand <= 0:
            return float('inf')

        days = inventory / daily_demand
        return days

    def _analyze_demand_trend(self, demand_history: np.ndarray) -> float:
        """
        分析需求趋势
        返回值：-1.0 (急剧下降) 到 1.0 (急剧上升)
        """
        if len(demand_history) < 14:
            return 0.0

        # 比较最近7天 vs 前7天
        recent = np.mean(demand_history[-7:])
        previous = np.mean(demand_history[-14:-7])

        if previous == 0:
            return 0.0

        trend = (recent - previous) / previous
        # 限制在 -1.0 到 1.0
        trend = max(-1.0, min(1.0, trend))

        return trend

    def _calculate_risk_score(self,
                             turnover: float,
                             days_supply: float,
                             age_days: int,
                             trend: float) -> float:
        """
        计算综合风险评分 (0-100)
        分数越高，风险越大
        """
        score = 0

        # 1. 周转率评分 (权重40%)
        if turnover < self.turnover_critical:
            score += 40
        elif turnover < self.turnover_high:
            score += 30
        elif turnover < self.turnover_medium:
            score += 15

        # 2. 供应天数评分 (权重30%)
        if days_supply > self.days_critical:
            score += 30
        elif days_supply > self.days_high:
            score += 20
        elif days_supply > self.days_medium:
            score += 10

        # 3. 库龄评分 (权重20%)
        age_score = min(20, (age_days / self.age_threshold) * 20)
        score += age_score

        # 4. 需求趋势评分 (权重10%)
        if trend < -0.3:  # 需求下降>30%
            score += 10
        elif trend < -0.1:  # 需求下降>10%
            score += 5

        return min(100, score)

    def _determine_risk_level(self,
                             risk_score: float,
                             turnover: float,
                             days_supply: float) -> DeadstockRiskLevel:
        """确定风险等级"""
        if risk_score >= 70 or turnover < self.turnover_critical or days_supply > self.days_critical:
            return DeadstockRiskLevel.CRITICAL
        elif risk_score >= 50 or turnover < self.turnover_high or days_supply > self.days_high:
            return DeadstockRiskLevel.HIGH
        elif risk_score >= 30 or turnover < self.turnover_medium or days_supply > self.days_medium:
            return DeadstockRiskLevel.MEDIUM
        elif risk_score >= 15:
            return DeadstockRiskLevel.LOW
        else:
            return DeadstockRiskLevel.HEALTHY

    def _estimate_potential_loss(self,
                                tied_capital: float,
                                age_days: int,
                                trend: float,
                                unit_cost: float,
                                unit_price: float) -> float:
        """估算潜在损失"""
        # 1. 资金机会成本 (假设年化回报15%)
        opportunity_cost = tied_capital * 0.15 * (age_days / 365)

        # 2. 持有成本 (仓储、保险等，年化24%)
        holding_cost = tied_capital * 0.24 * (age_days / 365)

        # 3. 贬值风险
        if trend < -0.2:  # 需求大幅下降
            depreciation_risk = tied_capital * 0.3  # 30%贬值风险
        elif trend < 0:
            depreciation_risk = tied_capital * 0.1  # 10%贬值风险
        else:
            depreciation_risk = 0

        total_loss = opportunity_cost + holding_cost + depreciation_risk
        return total_loss

    def _generate_action_plan(self,
                             risk_level: DeadstockRiskLevel,
                             inventory: float,
                             turnover: float,
                             days_supply: float,
                             unit_cost: float,
                             unit_price: float,
                             trend: float) -> Dict[str, Any]:
        """生成处理行动计划"""
        margin = unit_price - unit_cost
        margin_rate = margin / unit_price if unit_price > 0 else 0

        if risk_level == DeadstockRiskLevel.CRITICAL:
            # 高风险：立即清仓
            discount_rate = 0.25 if margin_rate > 0.4 else 0.15
            expected_recovery = inventory * unit_cost * (1 - discount_rate)

            return {
                'action': f'🚨 立即清仓促销 ({int(discount_rate * 100)}%折扣)',
                'details': {
                    'discount_rate': discount_rate,
                    'expected_price': unit_price * (1 - discount_rate),
                    'expected_recovery': expected_recovery,
                    'recovery_rate': (1 - discount_rate),
                    'timeline': '7-14天',
                    'alternative_1': '捐赠或销毁（税收抵扣）',
                    'alternative_2': '清仓给批发商',
                    'reasoning': f'库存天数{days_supply:.0f}天，严重积压，需立即释放资金'
                }
            }

        elif risk_level == DeadstockRiskLevel.HIGH:
            # 中高风险：促销清理
            discount_rate = 0.15
            expected_recovery = inventory * unit_cost * (1 - discount_rate * 0.7)

            return {
                'action': f'⚠️ 促销清理 ({int(discount_rate * 100)}%折扣)',
                'details': {
                    'discount_rate': discount_rate,
                    'expected_price': unit_price * (1 - discount_rate),
                    'expected_recovery': expected_recovery,
                    'recovery_rate': 0.9,
                    'timeline': '30天',
                    'bundling_option': '与畅销品捆绑销售',
                    'reasoning': f'周转率{turnover:.1f}次/年偏低，建议加速清理'
                }
            }

        elif risk_level == DeadstockRiskLevel.MEDIUM:
            # 中风险：减少订货+轻度促销
            reduction_pct = 0.5

            return {
                'action': '📉 减少补货 + 轻度促销 (5-10%折扣)',
                'details': {
                    'order_reduction': reduction_pct,
                    'discount_rate': 0.08,
                    'expected_clearance_days': 60,
                    'monitoring_frequency': '每周检查',
                    'reasoning': f'供应天数{days_supply:.0f}天偏高，需控制库存'
                }
            }

        elif risk_level == DeadstockRiskLevel.LOW:
            # 低风险：监控即可
            return {
                'action': '👀 持续监控，优化订货',
                'details': {
                    'order_adjustment': -0.2,
                    'review_frequency': '双周检查',
                    'reasoning': '库存健康，但需关注趋势变化'
                }
            }

        else:
            # 健康：保持现状
            return {
                'action': '✅ 库存健康，保持现状',
                'details': {
                    'reasoning': f'周转率{turnover:.1f}次/年良好，库存水平合理'
                }
            }

    def _calculate_urgency(self, risk_level: DeadstockRiskLevel, days_supply: float) -> int:
        """计算处理时限（天）"""
        if risk_level == DeadstockRiskLevel.CRITICAL:
            return 7
        elif risk_level == DeadstockRiskLevel.HIGH:
            return 30
        elif risk_level == DeadstockRiskLevel.MEDIUM:
            return 60
        elif risk_level == DeadstockRiskLevel.LOW:
            return 90
        else:
            return 0

    def scan_portfolio(self, inventory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        扫描整个库存组合，识别所有呆滞品

        Args:
            inventory_data: 所有SKU的库存数据列表

        Returns:
            扫描结果汇总
        """
        results = []
        risk_summary = {level: [] for level in DeadstockRiskLevel}
        total_tied_capital = 0
        total_recoverable = 0

        for item in inventory_data:
            analysis = self.analyze_sku(
                sku_id=item['sku_id'],
                current_inventory=item['current_inventory'],
                demand_history=item['demand_history'],
                unit_cost=item['unit_cost'],
                unit_price=item['unit_price'],
                inventory_age_days=item.get('age_days', 0)
            )

            results.append(analysis)
            risk_summary[analysis.risk_level].append(analysis)
            total_tied_capital += analysis.tied_up_capital

            if analysis.risk_level in [DeadstockRiskLevel.CRITICAL, DeadstockRiskLevel.HIGH]:
                # 估算可回收金额
                recovery_rate = analysis.action_details.get('recovery_rate', 0.7)
                total_recoverable += analysis.tied_up_capital * recovery_rate

        # 排序：风险从高到低
        risk_order = [DeadstockRiskLevel.CRITICAL, DeadstockRiskLevel.HIGH,
                     DeadstockRiskLevel.MEDIUM, DeadstockRiskLevel.LOW, DeadstockRiskLevel.HEALTHY]
        results.sort(key=lambda x: (risk_order.index(x.risk_level), -x.tied_up_capital))

        return {
            'analyses': results,
            'summary': {
                'total_skus_scanned': len(results),
                'critical_risk_count': len(risk_summary[DeadstockRiskLevel.CRITICAL]),
                'high_risk_count': len(risk_summary[DeadstockRiskLevel.HIGH]),
                'medium_risk_count': len(risk_summary[DeadstockRiskLevel.MEDIUM]),
                'total_tied_capital': total_tied_capital,
                'high_risk_capital': sum(a.tied_up_capital for a in risk_summary[DeadstockRiskLevel.CRITICAL] + risk_summary[DeadstockRiskLevel.HIGH]),
                'potential_recovery': total_recoverable,
                'estimated_annual_savings': total_recoverable * 0.24  # 释放资金节省的持有成本
            },
            'top_priorities': results[:10]  # 前10个最需要处理的
        }


if __name__ == "__main__":
    # 测试示例
    print("🔍 呆滞品智能识别系统 - 测试")
    print("=" * 70)

    detector = DeadstockDetector()

    # 模拟不同风险等级的商品
    test_cases = [
        {
            'name': '高风险呆滞品',
            'sku_id': 'SKU_DEAD_001',
            'current_inventory': 500,
            'demand_history': np.array([2, 1, 0, 1, 0, 0, 1] * 10),  # 极低需求
            'unit_cost': 20,
            'unit_price': 40,
            'age_days': 200
        },
        {
            'name': '健康商品',
            'sku_id': 'SKU_GOOD_001',
            'current_inventory': 100,
            'demand_history': np.array([45, 50, 48, 52, 49, 51, 47] * 10),  # 稳定高需求
            'unit_cost': 10,
            'unit_price': 20,
            'age_days': 15
        },
        {
            'name': '中风险商品',
            'sku_id': 'SKU_MED_001',
            'current_inventory': 300,
            'demand_history': np.array([10, 12, 11, 9, 10, 11, 10] * 10),  # 中等需求
            'unit_cost': 15,
            'unit_price': 30,
            'age_days': 60
        }
    ]

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"📦 测试: {case['name']}")
        print(f"SKU: {case['sku_id']}")

        analysis = detector.analyze_sku(
            sku_id=case['sku_id'],
            current_inventory=case['current_inventory'],
            demand_history=case['demand_history'],
            unit_cost=case['unit_cost'],
            unit_price=case['unit_price'],
            inventory_age_days=case['age_days']
        )

        print(f"\n🎯 分析结果:")
        print(f"风险等级: {analysis.risk_level.value.upper()}")
        print(f"风险评分: {analysis.risk_score:.1f}/100")
        print(f"周转率: {analysis.turnover_rate:.1f} 次/年")
        print(f"供应天数: {analysis.days_of_supply:.0f} 天")
        print(f"库龄: {analysis.inventory_age_days} 天")
        print(f"占用资金: ${analysis.tied_up_capital:,.0f}")
        print(f"潜在损失: ${analysis.potential_loss:,.0f}")
        print(f"\n💡 推荐行动: {analysis.recommended_action}")
        print(f"处理时限: {analysis.urgency_days} 天内")

        if analysis.action_details:
            print(f"\n📋 行动详情:")
            for key, value in analysis.action_details.items():
                if key != 'reasoning':
                    print(f"  • {key}: {value}")
            if 'reasoning' in analysis.action_details:
                print(f"\n  理由: {analysis.action_details['reasoning']}")

    print("\n" + "="*70)
    print("✅ 呆滞品检测测试完成！")
