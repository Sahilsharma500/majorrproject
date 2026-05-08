import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 24-Hour Grid Optimization Analysis\n",
                "This notebook generates 24 hours of environmental parameters, feeds them individually through the backend ML models, and passes the true predictive signals into the battery optimizer."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "# Ensure root project folder is pathable to pick up modules\n",
                "sys.path.append(os.getcwd())\n",
                "\n",
                "from prediction.load import predict_load\n",
                "from prediction.solar import predict_solar\n",
                "from prediction.wind import predict_wind\n",
                "from grid.cost_optimizer import run_grid_optimization\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import plotly.express as px\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Define 24-Hour Environmental Feature Data\n",
                "Here you can freely customize the hour-by-hour arrays to model specific weather patterns, solar conditions, and capacity constraints."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Base static configuration\n",
                "installed_solar_MW = 3000.0\n",
                "panel_area_m2 = 12750000.0\n",
                "num_panels = 7500000\n",
                "installed_wind_MW = 10.0\n",
                "\n",
                "# 24 arrays mapping specific hour conditions (Hour 0 to 23)\n",
                "hours = np.arange(24)\n",
                "times = [f'2025-10-13 {str(h).zfill(2)}:00:00' for h in hours]\n",
                "weathers = ['Sunny'] * 24\n",
                "temps_C = [22, 21, 20, 20, 21, 23, 25, 27, 30, 32, 34, 35, 36, 36, 35, 34, 31, 29, 27, 25, 24, 23, 23, 22]\n",
                "humidities = [80, 82, 85, 85, 83, 75, 65, 55, 45, 40, 38, 35, 35, 36, 38, 42, 50, 60, 68, 72, 75, 78, 79, 80]\n",
                "wind_speeds = [3.5, 3.2, 3.0, 3.1, 3.4, 4.0, 4.5, 5.0, 5.5, 6.0, 6.2, 6.0, 5.8, 5.5, 5.2, 4.8, 4.5, 3.8, 3.5, 3.2, 3.0, 3.1, 3.3, 3.4]\n",
                "precips = [0.0] * 24\n",
                "\n",
                "# Irradiance naturally rises at dawn and dies at dusk\n",
                "irradiances = [0, 0, 0, 0, 0, 0, 50, 200, 450, 650, 800, 950, 1000, 950, 750, 500, 250, 50, 0, 0, 0, 0, 0, 0]\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Connect 24-Hour Loop into ML Predictions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "solar_preds = []\n",
                "wind_preds = []\n",
                "load_preds = []\n",
                "\n",
                "for h in range(24):\n",
                "    load_inputs = {\n",
                "        'datetime': times[h], 'weather': weathers[h], 'temp_C': temps_C[h], \n",
                "        'humidity_%': humidities[h], 'wind_speed_m_s': wind_speeds[h], \n",
                "        'solar_irradiance_W_m2': irradiances[h], 'precip_mm': precips[h],\n",
                "        'installed_solar_MW': installed_solar_MW, 'panel_area_m2': panel_area_m2, \n",
                "        'num_panels': num_panels, 'installed_wind_MW': installed_wind_MW\n",
                "    }\n",
                "    \n",
                "    solar_inputs = {\n",
                "        'datetime': times[h], 'weather': weathers[h], 'temp_C': temps_C[h], \n",
                "        'humidity_%': humidities[h], 'wind_speed_m_s': wind_speeds[h], \n",
                "        'solar_irradiance_W_m2': irradiances[h], 'precip_mm': precips[h],\n",
                "        'installed_solar_MW': installed_solar_MW, 'panel_area_m2': panel_area_m2, 'num_panels': num_panels\n",
                "    }\n",
                "    \n",
                "    wind_inputs = {\n",
                "        'datetime': times[h], 'weather': weathers[h], 'temp_C': temps_C[h], \n",
                "        'humidity_%': humidities[h], 'wind_speed_m_s': wind_speeds[h], \n",
                "        'precip_mm': precips[h], 'installed_wind_MW': installed_wind_MW\n",
                "    }\n",
                "    \n",
                "    # Execute ML functions\n",
                "    load_preds.append(predict_load(load_inputs))\n",
                "    solar_preds.append(predict_solar(solar_inputs))\n",
                "    wind_preds.append(predict_wind(wind_inputs))\n",
                "\n",
                "load_profile = np.array(load_preds)\n",
                "solar_profile = np.array(solar_preds)\n",
                "wind_profile = np.array(wind_preds)\n",
                "\n",
                "print('24-Hour ML Predictions Extrapolated Successfully!')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Run Mathematical Grid Optimizer and Charting"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set your battery thresholds\n",
                "battery_capacity = 1000  # MWh\n",
                "max_power = 300  # MW\n",
                "\n",
                "res = run_grid_optimization(load_profile, solar_profile, wind_profile, battery_capacity, max_power)\n",
                "\n",
                "if res is None:\n",
                "    print('Error: Optimizer infeasible constraints.')\n",
                "else:\n",
                "    df = res['df']\n",
                "    \n",
                "    def format_inr(value):\n",
                "        abs_val = abs(value)\n",
                "        if abs_val >= 1e7:\n",
                "            return f'{value/1e7:,.2f} Cr'\n",
                "        elif abs_val >= 1e5:\n",
                "            return f'{value/1e5:,.2f} L'\n",
                "        else:\n",
                "            return f'{value:,.2f}'\n",
                "            \n",
                "    cost_str = f'Revenue: ₹ {format_inr(-res[\"total_cost\"])}' if res['total_cost'] < 0 else f'Cost: ₹ {format_inr(res[\"total_cost\"])}'\n",
                "    print(f'\\n--- Economic Performance ---')\n",
                "    print(f'Total {cost_str}')\n",
                "    print(f'Savings vs Grid-Only: ₹ {format_inr(res[\"savings\"])}')\n",
                "    print(f'Total Grid Import: {res[\"total_import\"]:,.2f} MWh')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize Dispatch Timeline\n",
                "chart_df = df[['Hour', 'Load (MW)', 'Solar (MW)', 'Wind (MW)', 'Grid Import (MW)']].set_index('Hour')\n",
                "fig1 = px.line(chart_df, title='True 24-Hour Power Grid Dispatch Timeline')\n",
                "fig1.show()\n",
                "\n",
                "# Battery SoC\n",
                "fig2 = px.line(df, x='Hour', y='SOC (%)', title='Battery Autonomy (%)')\n",
                "fig2.show()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Hourly Dispatch Data\n",
                "print('\\n--- Hourly Dispatch Data ---')\n",
                "display(df.style.highlight_max(axis=0))\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('24h_Optimization_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print("Notebook 24h_Optimization_Analysis.ipynb generated successfully!")
