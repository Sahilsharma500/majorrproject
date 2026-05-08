import numpy as np
import pandas as pd
import cvxpy as cp


def setup_tariffs():
    """
    Tamil Nadu TANGEDCO Time-of-Use pricing for commercial:
    - Peak hours (6-9 AM, 6-10 PM): ₹7.50/kWh -> (Hours 6,7,8 and 18,19,20,21)
    - Normal hours (9 AM-6 PM): ₹5.50/kWh -> (Hours 9 to 17)
    - Off-peak (night): ₹3.50/kWh -> (Hours 22, 23, 0, 1, 2, 3, 4, 5)
    - Net metering export tariff: ₹2.50/kWh
    """
    import_tariffs = np.zeros(24)
    for h in range(24):
        if h in [6, 7, 8, 18, 19, 20, 21]:
            import_tariffs[h] = 7.50
        elif 9 <= h <= 17:
            import_tariffs[h] = 5.50
        else:
            import_tariffs[h] = 3.50
            
    export_tariff = 2.50
    return import_tariffs * 1000, export_tariff * 1000

def run_grid_optimization(load_profile, solar_profile, wind_profile, battery_capacity=1000, max_power=300):
    horizon = 24
    import_tariffs, export_tariff = setup_tariffs()
    
    net_generation = solar_profile + wind_profile
    
    # Setup Decision Variables
    P_charge = cp.Variable(horizon)
    P_discharge = cp.Variable(horizon)
    P_grid_import = cp.Variable(horizon)
    P_grid_export = cp.Variable(horizon)
    SOC_var = cp.Variable(horizon + 1)
    
    # Battery parameters
    eta_c = 0.95
    eta_d = 0.95
    SOC_min = 10
    SOC_max = 90
    initial_SOC = 50
    dt = 1
    
    # Track the initial percentage
    constraints = [SOC_var[0] == initial_SOC]
    
    for k in range(horizon):
        # Battery Physics constraints
        constraints += [
            SOC_var[k+1] == SOC_var[k] + ((eta_c*P_charge[k]*dt - (P_discharge[k]*dt)/eta_d) / battery_capacity) * 100,
            SOC_var[k+1] >= SOC_min,
            SOC_var[k+1] <= SOC_max,
            P_charge[k] >= 0,
            P_charge[k] <= max_power,
            P_discharge[k] >= 0,
            P_discharge[k] <= max_power,
            P_grid_import[k] >= 0,
            P_grid_export[k] >= 0
        ]
        
        # Power matching constraint: The grid must perfectly balance the equation
        load_deficit = load_profile[k] - net_generation[k]
        constraints += [
            P_grid_import[k] - P_grid_export[k] == load_deficit + P_charge[k] - P_discharge[k]
        ]
        
    # Objective Function: Optimize Time-of-Use Arbitrage Costs
    # Minimize: Grid Purchases - Grid Sales
    cost = 0
    for k in range(horizon):
        # The true monetary cost
        cost += P_grid_import[k] * import_tariffs[k] - P_grid_export[k] * export_tariff
        
        # Soft penalty to prevent solver from uselessly charging and discharging at the exact same time
        cost += 1.0 * (P_charge[k] + P_discharge[k]) 
        
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)
    
    if problem.status in ["infeasible", "unbounded"] or P_charge.value is None:
        return None
        
    # Compute the purely theoretical naive baseline (no battery at all)
    baseline_imports = np.maximum(0, load_profile - net_generation)
    baseline_exports = np.maximum(0, net_generation - load_profile)
    baseline_cost_per_hour = baseline_imports * import_tariffs - baseline_exports * export_tariff
    baseline_total_cost = np.sum(baseline_cost_per_hour)
    
    opt_imports = P_grid_import.value
    opt_exports = P_grid_export.value
    opt_cost_per_hour = opt_imports * import_tariffs - opt_exports * export_tariff
    opt_total_cost = np.sum(opt_cost_per_hour)
    
    savings = baseline_total_cost - opt_total_cost
    total_grid_import = np.sum(opt_imports)
    
    df_results = pd.DataFrame({
        'Hour': np.arange(24),
        'Load (MW)': load_profile,
        'Solar (MW)': solar_profile,
        'Wind (MW)': wind_profile,
        'Net Load (MW)': load_profile - net_generation,
        'Battery Charge (MW)': P_charge.value,
        'Battery Discharge (MW)': P_discharge.value,
        'SOC (%)': SOC_var.value[:-1],
        'Grid Import (MW)': opt_imports,
        'Grid Export (MW)': opt_exports,
        'Grid Tariff (₹/MWh)': import_tariffs,
        'Hourly Cost (₹)': opt_cost_per_hour
    })
    
    df_results = df_results.round(2)
    
    return {
        'df': df_results,
        'total_cost': opt_total_cost,
        'baseline_cost': baseline_total_cost,
        'savings': savings,
        'total_import': total_grid_import
    }
