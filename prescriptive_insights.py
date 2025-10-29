import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PrescriptiveInsights:
    def __init__(self, data_path='.'):
        self.data_path = data_path
        
    def generate_maintenance_recommendations(self, predictions_df, risk_threshold=0.7):
        """Generate actionable maintenance recommendations"""
        
        recommendations = []
        
        for _, row in predictions_df.iterrows():
            inverter_id = row['inverter_id']
            failure_probability = row.get('failure_probability', 0)
            predicted_failure_type = row.get('predicted_failure_type', 'unknown')
            days_until_failure = row.get('days_until_failure', 30)
            
            if failure_probability >= risk_threshold:
                # High risk - immediate action required
                priority = 'HIGH'
                urgency_days = min(7, max(1, days_until_failure - 7))
                
                # Specific recommendations based on failure type
                if predicted_failure_type == 'overheating':
                    action = 'Inspect cooling system, clean air vents, check fan operation'
                    parts_needed = ['cooling_fan', 'thermal_paste']
                    estimated_cost = 400
                    
                elif predicted_failure_type == 'capacitor_failure':
                    action = 'Replace DC capacitors, test voltage levels'
                    parts_needed = ['dc_capacitor']
                    estimated_cost = 800
                    
                elif predicted_failure_type == 'fan_failure':
                    action = 'Replace cooling fan, check mounting'
                    parts_needed = ['cooling_fan']
                    estimated_cost = 300
                    
                elif predicted_failure_type == 'dc_connector_corrosion':
                    action = 'Clean and replace DC connectors, apply anti-corrosion treatment'
                    parts_needed = ['dc_connector', 'anti_corrosion_spray']
                    estimated_cost = 600
                    
                else:
                    action = 'Comprehensive inspection and diagnostic testing'
                    parts_needed = ['diagnostic_kit']
                    estimated_cost = 500
                    
            elif failure_probability >= 0.4:
                # Medium risk - scheduled maintenance
                priority = 'MEDIUM'
                urgency_days = min(14, days_until_failure - 7)
                action = 'Schedule preventive maintenance and detailed inspection'
                parts_needed = ['inspection_kit']
                estimated_cost = 200
                
            else:
                # Low risk - routine maintenance
                priority = 'LOW'
                urgency_days = 30
                action = 'Continue routine monitoring'
                parts_needed = []
                estimated_cost = 0
            
            # Additional recommendations based on environmental factors
            additional_actions = []
            
            # Check for soiling (cleaning recommendations)
            days_since_cleaning = row.get('days_since_cleaning', 0)
            if days_since_cleaning > 60:
                additional_actions.append('Schedule panel cleaning')
                estimated_cost += 100
            
            # Check for maintenance overdue
            days_since_maintenance = row.get('days_since_maintenance', 0)
            if days_since_maintenance > 180:
                additional_actions.append('Overdue maintenance - priority inspection required')
                estimated_cost += 150
            
            recommendations.append({
                'inverter_id': inverter_id,
                'priority': priority,
                'failure_probability': failure_probability,
                'predicted_failure_type': predicted_failure_type,
                'days_until_failure': days_until_failure,
                'recommended_action_by': datetime.now() + timedelta(days=urgency_days),
                'primary_action': action,
                'additional_actions': '; '.join(additional_actions) if additional_actions else 'None',
                'parts_needed': ', '.join(parts_needed) if parts_needed else 'None',
                'estimated_cost_usd': estimated_cost,
                'potential_savings_usd': self._calculate_potential_savings(failure_probability, predicted_failure_type)
            })
        
        return pd.DataFrame(recommendations)
    
    def _calculate_potential_savings(self, failure_prob, failure_type):
        """Calculate potential savings from preventive action"""
        
        # Cost of unplanned failure by type
        unplanned_costs = {
            'overheating': 2500,
            'capacitor_failure': 3000,
            'fan_failure': 1500,
            'dc_connector_corrosion': 2000,
            'control_board_failure': 4000,
            'unknown': 2500
        }
        
        # Downtime costs (revenue loss)
        downtime_hours = {
            'overheating': 12,
            'capacitor_failure': 24,
            'fan_failure': 8,
            'dc_connector_corrosion': 36,
            'control_board_failure': 72,
            'unknown': 24
        }
        
        unplanned_cost = unplanned_costs.get(failure_type, 2500)
        downtime_cost = downtime_hours.get(failure_type, 24) * 150  # $150/hour revenue loss
        
        total_avoided_cost = unplanned_cost + downtime_cost
        expected_savings = total_avoided_cost * failure_prob
        
        return round(expected_savings, 2)
    
    def optimize_maintenance_schedule(self, recommendations_df):
        """Optimize maintenance scheduling across the fleet"""
        
        # Group by priority and location (assuming all same location for simplicity)
        schedule = []
        
        # High priority - immediate scheduling
        high_priority = recommendations_df[recommendations_df['priority'] == 'HIGH'].copy()
        high_priority = high_priority.sort_values('failure_probability', ascending=False)
        
        # Medium priority - batch scheduling
        medium_priority = recommendations_df[recommendations_df['priority'] == 'MEDIUM'].copy()
        medium_priority = medium_priority.sort_values('recommended_action_by')
        
        # Create weekly schedule
        current_date = datetime.now()
        
        # Week 1: All high priority
        week1_capacity = 10  # Assume 10 inverters can be serviced per week
        week1_inverters = high_priority.head(week1_capacity)['inverter_id'].tolist()
        
        if len(week1_inverters) > 0:
            schedule.append({
                'week': 1,
                'start_date': current_date,
                'inverters': week1_inverters,
                'total_cost': high_priority.head(week1_capacity)['estimated_cost_usd'].sum(),
                'total_savings': high_priority.head(week1_capacity)['potential_savings_usd'].sum(),
                'priority_mix': 'HIGH'
            })
        
        # Week 2: Remaining high priority + medium priority
        remaining_high = high_priority.iloc[week1_capacity:] if len(high_priority) > week1_capacity else pd.DataFrame()
        week2_high_count = min(len(remaining_high), week1_capacity)
        week2_medium_count = week1_capacity - week2_high_count
        
        week2_inverters = []
        week2_cost = 0
        week2_savings = 0
        
        if week2_high_count > 0:
            week2_high = remaining_high.head(week2_high_count)
            week2_inverters.extend(week2_high['inverter_id'].tolist())
            week2_cost += week2_high['estimated_cost_usd'].sum()
            week2_savings += week2_high['potential_savings_usd'].sum()
        
        if week2_medium_count > 0:
            week2_medium = medium_priority.head(week2_medium_count)
            week2_inverters.extend(week2_medium['inverter_id'].tolist())
            week2_cost += week2_medium['estimated_cost_usd'].sum()
            week2_savings += week2_medium['potential_savings_usd'].sum()
        
        if len(week2_inverters) > 0:
            schedule.append({
                'week': 2,
                'start_date': current_date + timedelta(weeks=1),
                'inverters': week2_inverters,
                'total_cost': week2_cost,
                'total_savings': week2_savings,
                'priority_mix': 'HIGH + MEDIUM'
            })
        
        return pd.DataFrame(schedule)
    
    def calculate_business_impact(self, recommendations_df, current_om_cost_ratio=0.15):
        """Calculate quantified business impact"""
        
        # Current state analysis
        total_inverters = len(recommendations_df)
        high_risk_count = len(recommendations_df[recommendations_df['priority'] == 'HIGH'])
        medium_risk_count = len(recommendations_df[recommendations_df['priority'] == 'MEDIUM'])
        
        # Cost analysis
        total_preventive_cost = recommendations_df['estimated_cost_usd'].sum()
        total_potential_savings = recommendations_df['potential_savings_usd'].sum()
        net_benefit = total_potential_savings - total_preventive_cost
        
        # Downtime reduction calculation
        # Assume current 20% generation loss, target 5% with predictive maintenance
        current_generation_loss = 0.20
        target_generation_loss = 0.05
        downtime_reduction = current_generation_loss - target_generation_loss
        
        # Assume average 100kW per inverter, $0.10/kWh, 25% capacity factor
        avg_inverter_capacity = 100  # kW
        electricity_price = 0.10     # $/kWh
        capacity_factor = 0.25
        hours_per_year = 8760
        
        annual_generation_per_inverter = avg_inverter_capacity * hours_per_year * capacity_factor
        annual_revenue_per_inverter = annual_generation_per_inverter * electricity_price
        
        # Calculate impact
        current_loss_per_inverter = annual_revenue_per_inverter * current_generation_loss
        target_loss_per_inverter = annual_revenue_per_inverter * target_generation_loss
        annual_savings_per_inverter = current_loss_per_inverter - target_loss_per_inverter
        
        total_annual_revenue_recovery = annual_savings_per_inverter * total_inverters
        
        # O&M cost improvement
        current_om_cost = annual_revenue_per_inverter * current_om_cost_ratio * total_inverters
        target_om_cost_ratio = 0.08  # Target 8% vs current 15%
        target_om_cost = annual_revenue_per_inverter * target_om_cost_ratio * total_inverters
        om_cost_savings = current_om_cost - target_om_cost
        
        business_impact = {
            'total_inverters': total_inverters,
            'high_risk_inverters': high_risk_count,
            'medium_risk_inverters': medium_risk_count,
            'immediate_preventive_cost': total_preventive_cost,
            'immediate_potential_savings': total_potential_savings,
            'immediate_net_benefit': net_benefit,
            'annual_revenue_recovery': total_annual_revenue_recovery,
            'annual_om_cost_savings': om_cost_savings,
            'total_annual_benefit': total_annual_revenue_recovery + om_cost_savings,
            'downtime_reduction_percentage': downtime_reduction * 100,
            'om_cost_ratio_improvement': (current_om_cost_ratio - target_om_cost_ratio) * 100,
            'roi_percentage': (total_annual_revenue_recovery + om_cost_savings) / total_preventive_cost * 100 if total_preventive_cost > 0 else 0
        }
        
        return business_impact
    
    def generate_executive_summary(self, recommendations_df):
        """Generate executive summary with key insights"""
        
        business_impact = self.calculate_business_impact(recommendations_df)
        schedule_df = self.optimize_maintenance_schedule(recommendations_df)
        
        print("=== PREDICTIVE MAINTENANCE EXECUTIVE SUMMARY ===\n")
        
        print("1. CURRENT FLEET STATUS")
        print(f"Total Inverters Analyzed: {business_impact['total_inverters']}")
        print(f"High Risk Inverters: {business_impact['high_risk_inverters']} ({business_impact['high_risk_inverters']/business_impact['total_inverters']*100:.1f}%)")
        print(f"Medium Risk Inverters: {business_impact['medium_risk_inverters']} ({business_impact['medium_risk_inverters']/business_impact['total_inverters']*100:.1f}%)")
        
        print(f"\n2. IMMEDIATE ACTIONS REQUIRED")
        print(f"Preventive Maintenance Cost: ${business_impact['immediate_preventive_cost']:,.2f}")
        print(f"Potential Failure Cost Avoided: ${business_impact['immediate_potential_savings']:,.2f}")
        print(f"Immediate Net Benefit: ${business_impact['immediate_net_benefit']:,.2f}")
        
        print(f"\n3. ANNUAL BUSINESS IMPACT")
        print(f"Generation Loss Reduction: {business_impact['downtime_reduction_percentage']:.1f}% (from 20% to 5%)")
        print(f"Annual Revenue Recovery: ${business_impact['annual_revenue_recovery']:,.2f}")
        print(f"O&M Cost Ratio Improvement: {business_impact['om_cost_ratio_improvement']:.1f}% (from 15% to 8%)")
        print(f"Annual O&M Cost Savings: ${business_impact['annual_om_cost_savings']:,.2f}")
        print(f"Total Annual Benefit: ${business_impact['total_annual_benefit']:,.2f}")
        print(f"Return on Investment: {business_impact['roi_percentage']:.1f}%")
        
        print(f"\n4. RECOMMENDED IMPLEMENTATION SCHEDULE")
        for _, week in schedule_df.iterrows():
            print(f"Week {week['week']}: {len(week['inverters'])} inverters ({week['priority_mix']})")
            print(f"  Cost: ${week['total_cost']:,.2f}, Savings: ${week['total_savings']:,.2f}")
        
        print(f"\n5. KEY RECOMMENDATIONS")
        recommendations = [
            "Implement predictive maintenance system immediately for high-risk inverters",
            "Establish 2-week maintenance scheduling based on failure probability rankings",
            "Deploy automated monitoring and alerting system",
            "Train maintenance team on predictive maintenance protocols",
            "Set up monthly model performance review and recalibration",
            "Integrate with existing CMMS (Computerized Maintenance Management System)"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return business_impact, schedule_df

if __name__ == "__main__":
    # Create sample predictions for demonstration
    np.random.seed(42)
    n_inverters = 50
    
    sample_predictions = pd.DataFrame({
        'inverter_id': [f'INV_{i:03d}' for i in range(1, n_inverters + 1)],
        'failure_probability': np.random.beta(2, 8, n_inverters),
        'predicted_failure_type': np.random.choice([
            'overheating', 'capacitor_failure', 'fan_failure', 
            'dc_connector_corrosion', 'control_board_failure'
        ], n_inverters),
        'days_until_failure': np.random.randint(7, 35, n_inverters),
        'days_since_cleaning': np.random.randint(0, 120, n_inverters),
        'days_since_maintenance': np.random.randint(0, 300, n_inverters)
    })
    
    prescriptive = PrescriptiveInsights()
    
    # Generate recommendations
    recommendations_df = prescriptive.generate_maintenance_recommendations(sample_predictions)
    
    # Generate executive summary
    business_impact, schedule_df = prescriptive.generate_executive_summary(recommendations_df)
    
    # Save results
    recommendations_df.to_csv('/mnt/e/Technical Test/Logicode - Sinarmas Multiartha/AWS/maintenance_recommendations.csv', index=False)
    schedule_df.to_csv('/mnt/e/Technical Test/Logicode - Sinarmas Multiartha/AWS/maintenance_schedule.csv', index=False)
    
    print(f"\nRecommendations saved to maintenance_recommendations.csv")
    print(f"Schedule saved to maintenance_schedule.csv")
