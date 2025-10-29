import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SolarPVDataExplorer:
    def __init__(self, data_path='.'):
        self.data_path = data_path
        self.performance_df = None
        self.maintenance_df = None
        self.failures_df = None
        self.parts_df = None
        self.financial_df = None
        
    def load_data(self):
        """Load all datasets"""
        try:
            self.performance_df = pd.read_csv('performance_data.csv')
            self.maintenance_df = pd.read_csv('maintenance_logs.csv')
            self.failures_df = pd.read_csv('failure_records.csv')
            self.parts_df = pd.read_csv('spare_parts.csv')
            self.financial_df = pd.read_csv('financial_data.csv')
            
            # Convert datetime columns
            self.performance_df['timestamp'] = pd.to_datetime(self.performance_df['timestamp'])
            self.maintenance_df['service_date'] = pd.to_datetime(self.maintenance_df['service_date'])
            self.failures_df['failure_start'] = pd.to_datetime(self.failures_df['failure_start'])
            self.failures_df['failure_end'] = pd.to_datetime(self.failures_df['failure_end'])
            self.parts_df['install_date'] = pd.to_datetime(self.parts_df['install_date'])
            self.parts_df['replacement_date'] = pd.to_datetime(self.parts_df['replacement_date'])
            
            print("Data loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def identify_data_quality_issues(self):
        """Identify gaps, anomalies, and data quality issues"""
        issues = {}
        
        # Performance data issues
        perf_issues = []
        
        # Missing values
        missing_vals = self.performance_df.isnull().sum()
        if missing_vals.sum() > 0:
            perf_issues.append(f"Missing values: {missing_vals[missing_vals > 0].to_dict()}")
        
        # Negative energy generation
        negative_gen = (self.performance_df['energy_kwh'] < 0).sum()
        if negative_gen > 0:
            perf_issues.append(f"Negative energy generation: {negative_gen} records")
        
        # Unrealistic irradiance values
        high_irradiance = (self.performance_df['irradiance_wm2'] > 1200).sum()
        if high_irradiance > 0:
            perf_issues.append(f"Unrealistic high irradiance (>1200 W/m²): {high_irradiance} records")
        
        # Temperature anomalies
        extreme_temp = ((self.performance_df['module_temp_c'] > 85) | 
                       (self.performance_df['module_temp_c'] < -10)).sum()
        if extreme_temp > 0:
            perf_issues.append(f"Extreme temperatures: {extreme_temp} records")
        
        # Generation during zero irradiance
        zero_irr_gen = ((self.performance_df['irradiance_wm2'] == 0) & 
                       (self.performance_df['energy_kwh'] > 0)).sum()
        if zero_irr_gen > 0:
            perf_issues.append(f"Generation during zero irradiance: {zero_irr_gen} records")
        
        issues['performance'] = perf_issues
        
        # Maintenance data issues
        maint_issues = []
        
        # Future maintenance dates
        future_maint = (self.maintenance_df['service_date'] > datetime.now()).sum()
        if future_maint > 0:
            maint_issues.append(f"Future maintenance dates: {future_maint} records")
        
        # Negative costs
        negative_cost = (self.maintenance_df['cost_usd'] < 0).sum()
        if negative_cost > 0:
            maint_issues.append(f"Negative maintenance costs: {negative_cost} records")
        
        issues['maintenance'] = maint_issues
        
        # Failure data issues
        fail_issues = []
        
        # Failure end before start
        invalid_duration = (self.failures_df['failure_end'] < self.failures_df['failure_start']).sum()
        if invalid_duration > 0:
            fail_issues.append(f"Invalid failure duration: {invalid_duration} records")
        
        # Extremely long downtimes (>30 days)
        long_downtime = (self.failures_df['downtime_hours'] > 720).sum()
        if long_downtime > 0:
            fail_issues.append(f"Extremely long downtimes (>30 days): {long_downtime} records")
        
        issues['failures'] = fail_issues
        
        return issues
    
    def detect_outliers(self):
        """Detect outliers using statistical methods"""
        outliers = {}
        
        # Performance outliers using IQR method
        numeric_cols = ['energy_kwh', 'irradiance_wm2', 'module_temp_c', 'dc_power_kw', 'ac_power_kw']
        
        for col in numeric_cols:
            Q1 = self.performance_df[col].quantile(0.25)
            Q3 = self.performance_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.performance_df[col] < lower_bound) | 
                           (self.performance_df[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.performance_df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
        
        return outliers
    
    def analyze_data_gaps(self):
        """Analyze temporal gaps in data"""
        gaps = {}
        
        # Check for missing time intervals in performance data
        for inverter in self.performance_df['inverter_id'].unique()[:5]:  # Check first 5 inverters
            inv_data = self.performance_df[self.performance_df['inverter_id'] == inverter].copy()
            inv_data = inv_data.sort_values('timestamp')
            
            # Expected 5-minute intervals
            expected_intervals = pd.date_range(
                start=inv_data['timestamp'].min(),
                end=inv_data['timestamp'].max(),
                freq='5min'
            )
            
            missing_intervals = len(expected_intervals) - len(inv_data)
            if missing_intervals > 0:
                gaps[inverter] = {
                    'missing_intervals': missing_intervals,
                    'data_completeness': len(inv_data) / len(expected_intervals) * 100
                }
        
        return gaps
    
    def propose_data_reconciliation(self):
        """Propose methods for data reconciliation"""
        recommendations = {
            'performance_data': [
                "Interpolate missing 5-minute intervals using linear interpolation for irradiance and temperature",
                "Set energy generation to 0 for nighttime missing intervals (irradiance = 0)",
                "Flag and investigate negative energy values - likely sensor errors",
                "Cap irradiance values at 1200 W/m² (theoretical maximum)",
                "Apply temperature bounds: -10°C to 85°C for module temperature"
            ],
            'maintenance_data': [
                "Validate maintenance dates against asset installation dates",
                "Cross-reference part replacement with failure records",
                "Standardize maintenance event types and part names",
                "Verify cost data against industry benchmarks"
            ],
            'failure_data': [
                "Ensure failure_end > failure_start for all records",
                "Cross-validate downtime hours with timestamp differences",
                "Map fault codes to standardized failure categories",
                "Link failure records with corresponding maintenance actions"
            ],
            'data_integration': [
                "Create master asset registry with inverter specifications",
                "Establish data quality rules and automated validation",
                "Implement real-time data monitoring and alerting",
                "Set up regular data reconciliation processes"
            ]
        }
        
        return recommendations
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("=== SOLAR PV DATA QUALITY ASSESSMENT ===\n")
        
        # Dataset overview
        print("1. DATASET OVERVIEW")
        print(f"Performance records: {len(self.performance_df):,}")
        print(f"Maintenance records: {len(self.maintenance_df):,}")
        print(f"Failure records: {len(self.failures_df):,}")
        print(f"Spare parts records: {len(self.parts_df):,}")
        print(f"Financial records: {len(self.financial_df):,}")
        print(f"Date range: {self.performance_df['timestamp'].min()} to {self.performance_df['timestamp'].max()}")
        print(f"Number of inverters: {self.performance_df['inverter_id'].nunique()}\n")
        
        # Data quality issues
        print("2. DATA QUALITY ISSUES")
        issues = self.identify_data_quality_issues()
        for category, issue_list in issues.items():
            print(f"\n{category.upper()}:")
            if issue_list:
                for issue in issue_list:
                    print(f"  - {issue}")
            else:
                print("  - No issues detected")
        
        # Outliers
        print("\n3. OUTLIER ANALYSIS")
        outliers = self.detect_outliers()
        if outliers:
            for col, stats in outliers.items():
                print(f"{col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        else:
            print("No significant outliers detected")
        
        # Data gaps
        print("\n4. DATA COMPLETENESS")
        gaps = self.analyze_data_gaps()
        if gaps:
            for inverter, gap_info in gaps.items():
                print(f"{inverter}: {gap_info['data_completeness']:.1f}% complete")
        else:
            print("Data appears complete for sampled inverters")
        
        # Recommendations
        print("\n5. RECONCILIATION RECOMMENDATIONS")
        recommendations = self.propose_data_reconciliation()
        for category, rec_list in recommendations.items():
            print(f"\n{category.replace('_', ' ').upper()}:")
            for rec in rec_list:
                print(f"  - {rec}")

if __name__ == "__main__":
    explorer = SolarPVDataExplorer()
    if explorer.load_data():
        explorer.generate_data_quality_report()
