import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class RealisticSolarPVDataGenerator:
    def __init__(self, start_date='2023-01-01', end_date='2023-04-01', n_inverters=20):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.n_inverters = n_inverters
        
    def generate_performance_data(self):
        """Generate realistic 5-minute performance data based on actual solar patterns"""
        date_range = pd.date_range(self.start_date, self.end_date, freq='5min')
        
        data = []
        row_count = 0
        max_rows = 500000  # Limit to exactly 500k rows
        
        for inverter_id in range(1, self.n_inverters + 1):
            if row_count >= max_rows:
                break
                
            # Each inverter has different characteristics
            inverter_capacity = random.uniform(100, 500)  # kW capacity
            degradation_rate = random.uniform(0.5, 0.8)  # % per year
            soiling_factor = 1.0
            last_cleaning = self.start_date
            
            for timestamp in date_range:
                if row_count >= max_rows:
                    break
                    
                hour = timestamp.hour
                day_of_year = timestamp.dayofyear
                
                # Realistic solar irradiance pattern (Indonesia tropical)
                if 6 <= hour <= 18:
                    # Peak irradiance around 1000 W/m2 at noon
                    solar_angle = np.sin(np.pi * (hour - 6) / 12)
                    base_irradiance = 1000 * solar_angle
                    
                    # Seasonal variation (less pronounced in tropics)
                    seasonal_factor = 0.9 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
                    
                    # Weather variations (cloudy days, rain)
                    weather_rand = random.random()
                    if weather_rand < 0.15:  # 15% very cloudy
                        weather_factor = random.uniform(0.1, 0.3)
                    elif weather_rand < 0.35:  # 20% partly cloudy
                        weather_factor = random.uniform(0.4, 0.7)
                    else:  # 65% clear/sunny
                        weather_factor = random.uniform(0.8, 1.0)
                    
                    irradiance = base_irradiance * seasonal_factor * weather_factor
                else:
                    irradiance = 0
                
                # Module temperature (realistic for tropical climate)
                ambient_temp = 27 + 5 * np.sin(2 * np.pi * day_of_year / 365) + 3 * np.sin(2 * np.pi * hour / 24)
                if irradiance > 0:
                    module_temp = ambient_temp + (irradiance / 1000) * 30  # Cell temp rise
                else:
                    module_temp = ambient_temp
                
                # Soiling accumulation (reduces performance over time)
                days_since_cleaning = (timestamp - last_cleaning).days
                if days_since_cleaning > random.randint(30, 90):  # Random cleaning schedule
                    last_cleaning = timestamp
                    soiling_factor = 1.0
                else:
                    # Soiling reduces performance by 0.2% per day
                    soiling_factor = max(0.7, 1.0 - (days_since_cleaning * 0.002))
                
                # Performance degradation over time
                years_operating = (timestamp - self.start_date).days / 365
                degradation_factor = 1 - (degradation_rate / 100 * years_operating)
                
                # Temperature coefficient effect (-0.4% per °C above 25°C)
                temp_coefficient = 1 - max(0, (module_temp - 25) * 0.004)
                
                # Energy generation calculation
                if irradiance > 50:  # Minimum irradiance for generation
                    dc_power = inverter_capacity * (irradiance / 1000) * degradation_factor * soiling_factor * temp_coefficient
                    # Inverter efficiency (realistic curve)
                    if dc_power < inverter_capacity * 0.1:
                        inv_efficiency = 0.85
                    elif dc_power < inverter_capacity * 0.5:
                        inv_efficiency = 0.95
                    else:
                        inv_efficiency = 0.97
                    
                    ac_power = dc_power * inv_efficiency
                    energy_kwh = ac_power * (5/60)  # 5-minute interval
                else:
                    energy_kwh = 0
                
                # Fault simulation based on real failure modes
                fault_code = 0
                inverter_status = 'NORMAL'
                
                # Higher fault probability during high temperature
                base_fault_prob = 0.0001
                if module_temp > 70:
                    fault_prob = base_fault_prob * 5
                elif module_temp > 60:
                    fault_prob = base_fault_prob * 2
                else:
                    fault_prob = base_fault_prob
                
                if random.random() < fault_prob:
                    fault_codes = {
                        101: 'DC_OVERVOLTAGE',
                        102: 'DC_UNDERVOLTAGE', 
                        201: 'AC_OVERVOLTAGE',
                        202: 'AC_UNDERVOLTAGE',
                        301: 'OVERTEMPERATURE',
                        302: 'FAN_FAILURE',
                        401: 'INSULATION_FAULT',
                        501: 'GRID_FAULT'
                    }
                    fault_code = random.choice(list(fault_codes.keys()))
                    inverter_status = 'FAULT'
                    energy_kwh = 0  # No generation during fault
                
                data.append({
                    'timestamp': timestamp,
                    'inverter_id': f'INV_{inverter_id:03d}',
                    'energy_kwh': round(max(0, energy_kwh), 3),
                    'irradiance_wm2': round(max(0, irradiance), 1),
                    'module_temp_c': round(module_temp, 1),
                    'ambient_temp_c': round(ambient_temp, 1),
                    'inverter_status': inverter_status,
                    'fault_code': fault_code,
                    'dc_power_kw': round(max(0, dc_power if 'dc_power' in locals() else 0), 2),
                    'ac_power_kw': round(max(0, ac_power if 'ac_power' in locals() else 0), 2)
                })
                row_count += 1
        
        return pd.DataFrame(data)
    
    def generate_maintenance_data(self):
        """Generate realistic maintenance and failure data"""
        maintenance_logs = []
        failure_records = []
        spare_parts = []
        
        # Real failure causes and their frequencies
        failure_causes = {
            'overheating': 0.25,
            'capacitor_failure': 0.20,
            'fan_failure': 0.15,
            'dc_connector_corrosion': 0.12,
            'control_board_failure': 0.10,
            'fuse_blown': 0.08,
            'insulation_degradation': 0.06,
            'grid_disturbance': 0.04
        }
        
        for inverter_id in range(1, self.n_inverters + 1):
            inverter_name = f'INV_{inverter_id:03d}'
            
            # Spare parts lifecycle - initial installation
            part_types = ['capacitor', 'cooling_fan', 'dc_fuse', 'control_board', 'contactor', 'surge_protector']
            for part_type in part_types:
                install_date = self.start_date + timedelta(days=random.randint(-365, 0))  # Some parts pre-installed
                
                # Expected lifespan based on part type (in days)
                expected_lifespan = {
                    'capacitor': random.randint(2555, 3650),  # 7-10 years
                    'cooling_fan': random.randint(1825, 2555),  # 5-7 years
                    'dc_fuse': random.randint(3650, 5475),  # 10-15 years
                    'control_board': random.randint(2920, 4380),  # 8-12 years
                    'contactor': random.randint(1460, 2190),  # 4-6 years
                    'surge_protector': random.randint(1825, 2920)  # 5-8 years
                }[part_type]
                
                # Actual lifespan varies ±30% from expected
                actual_lifespan = int(expected_lifespan * random.uniform(0.7, 1.3))
                
                spare_parts.append({
                    'inverter_id': inverter_name,
                    'part_type': part_type,
                    'part_serial': f'{part_type.upper()}_{inverter_id}_{random.randint(1000,9999)}',
                    'install_date': install_date,
                    'expected_lifespan_days': expected_lifespan,
                    'actual_lifespan_days': actual_lifespan,
                    'replacement_date': install_date + timedelta(days=actual_lifespan),
                    'status': 'active' if (install_date + timedelta(days=actual_lifespan)) > datetime.now() else 'replaced',
                    'cost_usd': {
                        'capacitor': random.randint(200, 500),
                        'cooling_fan': random.randint(150, 350),
                        'dc_fuse': random.randint(50, 150),
                        'control_board': random.randint(800, 1500),
                        'contactor': random.randint(100, 300),
                        'surge_protector': random.randint(80, 200)
                    }[part_type]
                })
            
            # Preventive maintenance (quarterly cleaning, annual inspection)
            current_date = self.start_date
            while current_date < self.end_date:
                # Quarterly cleaning
                if current_date.month % 3 == 1:
                    maintenance_logs.append({
                        'service_date': current_date + timedelta(days=random.randint(0, 15)),
                        'inverter_id': inverter_name,
                        'event_type': 'cleaning',
                        'parts_replaced': 'none',
                        'cost_usd': random.randint(50, 150),
                        'planned': True,
                        'technician_hours': random.uniform(1, 3)
                    })
                
                # Annual inspection
                if current_date.month == 1:
                    maintenance_logs.append({
                        'service_date': current_date + timedelta(days=random.randint(0, 30)),
                        'inverter_id': inverter_name,
                        'event_type': 'inspection',
                        'parts_replaced': 'none',
                        'cost_usd': random.randint(200, 400),
                        'planned': True,
                        'technician_hours': random.uniform(2, 4)
                    })
                
                current_date += timedelta(days=90)
            
            # Corrective maintenance (failures)
            n_failures = np.random.poisson(0.5)  # Reduced for shorter timeframe (3 months vs 18 months)
            
            for _ in range(n_failures):
                failure_start = self.start_date + timedelta(days=random.randint(0, 90))
                
                # Select failure cause based on realistic probabilities
                cause = np.random.choice(list(failure_causes.keys()), 
                                       p=list(failure_causes.values()))
                
                # Downtime varies by failure type
                downtime_map = {
                    'overheating': (4, 24),
                    'capacitor_failure': (8, 48),
                    'fan_failure': (6, 36),
                    'dc_connector_corrosion': (12, 72),
                    'control_board_failure': (24, 120),
                    'fuse_blown': (2, 8),
                    'insulation_degradation': (48, 168),
                    'grid_disturbance': (1, 6)
                }
                
                min_hours, max_hours = downtime_map[cause]
                downtime_hours = random.randint(min_hours, max_hours)
                failure_end = failure_start + timedelta(hours=downtime_hours)
                
                # Resolution and cost based on cause
                resolution_map = {
                    'overheating': ('fan_replacement', random.randint(300, 800)),
                    'capacitor_failure': ('capacitor_replacement', random.randint(500, 1200)),
                    'fan_failure': ('fan_replacement', random.randint(200, 600)),
                    'dc_connector_corrosion': ('connector_replacement', random.randint(400, 1000)),
                    'control_board_failure': ('board_replacement', random.randint(1500, 3000)),
                    'fuse_blown': ('fuse_replacement', random.randint(50, 200)),
                    'insulation_degradation': ('cable_replacement', random.randint(800, 2000)),
                    'grid_disturbance': ('reset_protection', random.randint(100, 300))
                }
                
                resolution, cost = resolution_map[cause]
                
                failure_records.append({
                    'failure_start': failure_start,
                    'failure_end': failure_end,
                    'inverter_id': inverter_name,
                    'cause': cause,
                    'resolution': resolution,
                    'downtime_hours': downtime_hours,
                    'repair_cost_usd': cost,
                    'revenue_loss_usd': downtime_hours * random.randint(80, 150),  # Lost revenue
                    'mttr_hours': downtime_hours,  # Mean Time To Repair
                    'severity': 'high' if downtime_hours > 48 else 'medium' if downtime_hours > 12 else 'low'
                })
                
                # Add corrective maintenance record
                maintenance_logs.append({
                    'service_date': failure_end,
                    'inverter_id': inverter_name,
                    'event_type': 'repair',
                    'parts_replaced': resolution.split('_')[0],
                    'cost_usd': cost,
                    'planned': False,
                    'technician_hours': downtime_hours * 0.6  # Actual work time vs total downtime
                })
        
        return pd.DataFrame(maintenance_logs), pd.DataFrame(failure_records), pd.DataFrame(spare_parts)
    
    def generate_financial_data(self):
        """Generate comprehensive financial impact data"""
        financial_data = []
        
        for inverter_id in range(1, self.n_inverters + 1):
            inverter_name = f'INV_{inverter_id:03d}'
            inverter_capacity = random.uniform(100, 500)  # kW
            
            # Monthly financial tracking
            current_date = self.start_date
            while current_date < self.end_date:
                month_str = current_date.strftime('%Y-%m')
                
                # Expected monthly generation (kWh)
                expected_generation = inverter_capacity * 24 * 30 * 0.25  # 25% capacity factor
                
                # Actual generation (with losses)
                generation_loss_factor = random.uniform(0.75, 0.95)  # 5-25% loss
                actual_generation = expected_generation * generation_loss_factor
                
                # Revenue calculation ($/kWh)
                electricity_price = random.uniform(0.08, 0.12)  # USD per kWh
                expected_revenue = expected_generation * electricity_price
                actual_revenue = actual_generation * electricity_price
                revenue_loss = expected_revenue - actual_revenue
                
                # O&M costs
                planned_om_cost = random.randint(200, 600)  # Monthly planned maintenance
                unplanned_om_cost = random.randint(0, 2000)  # Unplanned repairs
                
                # Performance ratio
                performance_ratio = actual_generation / expected_generation
                
                financial_data.append({
                    'month': month_str,
                    'inverter_id': inverter_name,
                    'expected_generation_kwh': round(expected_generation, 2),
                    'actual_generation_kwh': round(actual_generation, 2),
                    'generation_loss_kwh': round(expected_generation - actual_generation, 2),
                    'electricity_price_usd_kwh': round(electricity_price, 4),
                    'expected_revenue_usd': round(expected_revenue, 2),
                    'actual_revenue_usd': round(actual_revenue, 2),
                    'revenue_loss_usd': round(revenue_loss, 2),
                    'planned_om_cost_usd': planned_om_cost,
                    'unplanned_om_cost_usd': unplanned_om_cost,
                    'total_om_cost_usd': planned_om_cost + unplanned_om_cost,
                    'performance_ratio': round(performance_ratio, 3),
                    'om_cost_ratio': round((planned_om_cost + unplanned_om_cost) / actual_revenue, 3)
                })
                
                current_date += timedelta(days=30)
        
        return pd.DataFrame(financial_data)

if __name__ == "__main__":
    generator = RealisticSolarPVDataGenerator()
    
    print("Generating realistic performance data...")
    performance_df = generator.generate_performance_data()
    performance_df.to_csv('performance_data.csv', index=False)
    print(f"Generated {len(performance_df)} performance records")
    
    print("Generating realistic maintenance data...")
    maintenance_df, failures_df, parts_df = generator.generate_maintenance_data()
    maintenance_df.to_csv('maintenance_logs.csv', index=False)
    failures_df.to_csv('failure_records.csv', index=False)
    parts_df.to_csv('spare_parts.csv', index=False)
    print(f"Generated {len(maintenance_df)} maintenance records, {len(failures_df)} failure records, {len(parts_df)} spare parts records")
    
    print("Generating financial data...")
    financial_df = generator.generate_financial_data()
    financial_df.to_csv('financial_data.csv', index=False)
    print(f"Generated {len(financial_df)} financial records")
    
    print("Data generation complete!")
