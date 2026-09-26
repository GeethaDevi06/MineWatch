"""
MineWatch V2 - Synthetic Dataset Generator
============================================
Generates a realistic, reproducible synthetic mining dataset for the MineWatch
EDA project (inspired by SIH26024 - Coal Mine Governance & Compliance).

IMPORTANT: This dataset is entirely synthetic / project-generated. It does not
represent real mines, real companies, or real government records. It is built
to have realistic *relationships* between variables (production drives energy,
waste, revenue, etc.) and to intentionally contain data-quality issues
(missing values, duplicates, inconsistent category labels, outliers) so that
the accompanying EDA notebook has genuine cleaning work to demonstrate.

Run:
    python generate_dataset.py

Output:
    minewatch_data.csv  (~3,272 rows x ~34 columns)
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0. REPRODUCIBILITY
# ------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. MINE MASTER DATA (30 fictional mines)
# ------------------------------------------------------------------
N_MINES = 30

REGIONS_STATES = [
    ("Eastern", "Jharkhand"), ("Eastern", "West Bengal"), ("Eastern", "Odisha"),
    ("Central", "Chhattisgarh"), ("Central", "Madhya Pradesh"),
    ("Western", "Maharashtra"), ("Western", "Gujarat"),
    ("Southern", "Telangana"), ("Southern", "Andhra Pradesh"),
    ("Northern", "Uttar Pradesh"),
]

MINE_TYPES = ["Open Cast", "Underground", "Mixed"]
COMMODITIES = ["Coal", "Iron Ore", "Bauxite", "Limestone", "Copper Ore"]

# Inconsistent-label variants used later to dirty a slice of the data
MINE_TYPE_VARIANTS = {
    "Open Cast": ["Open Cast", "Open-Cast", "open cast", "OPEN CAST"],
    "Underground": ["Underground", "underground", "UNDERGROUND"],
    "Mixed": ["Mixed", "mixed", "MIXED"],
}

mines = []
for i in range(1, N_MINES + 1):
    mine_id = f"MINE-{i:03d}"
    region, state = REGIONS_STATES[(i - 1) % len(REGIONS_STATES)]
    mine_type = rng.choice(MINE_TYPES, p=[0.5, 0.3, 0.2])
    commodity = rng.choice(COMMODITIES, p=[0.45, 0.2, 0.15, 0.12, 0.08])

    # Mine "scale" drives baseline production, workforce, reserves etc.
    scale = rng.lognormal(mean=0.0, sigma=0.55)  # relative size multiplier
    base_production = rng.uniform(8_000, 60_000) * scale       # tonnes/month baseline
    initial_reserve = base_production * rng.uniform(220, 420)   # ~18-35 years of reserve
    base_workers = max(50, int(rng.uniform(150, 1400) * scale))

    # Each mine gets its own long-run safety "personality"
    safety_trend = rng.choice(["increasing", "decreasing", "stable", "spiky"],
                               p=[0.2, 0.3, 0.35, 0.15])
    base_incident_rate = rng.uniform(0.3, 3.0)  # incidents per month baseline

    mines.append({
        "Mine_ID": mine_id,
        "Mine_Name": f"{state.split()[0]} {commodity.split()[0]} Mine {i}",
        "Region": region,
        "State": state,
        "Mine_Type": mine_type,
        "Commodity": commodity,
        "_scale": scale,
        "_base_production": base_production,
        "_initial_reserve": initial_reserve,
        "_base_workers": base_workers,
        "_safety_trend": safety_trend,
        "_base_incident_rate": base_incident_rate,
    })

mines_df = pd.DataFrame(mines)

# ------------------------------------------------------------------
# 2. TIME INDEX (2016-01 .. 2024-12, monthly)
# ------------------------------------------------------------------
years = list(range(2016, 2025))
months = list(range(1, 13))

VIOLATION_CATEGORIES = ["Safety", "Environmental", "Operational",
                         "Equipment", "Documentation", "Emergency Preparedness"]
INCIDENT_CATEGORIES = ["Fall of Roof/Rock", "Machinery", "Haulage/Transport",
                        "Fire", "Gas/Explosion", "Electrical", "Slip/Fall"]
SEVERITIES = ["Minor", "Moderate", "Serious", "Fatal"]
SEVERITY_P = [0.62, 0.26, 0.10, 0.02]
ACTION_STATUS = ["Completed", "Pending", "In Progress", "Overdue"]

# Commodity price base levels (arbitrary indexed units) + yearly drift/noise
commodity_base_price = {"Coal": 2400, "Iron Ore": 5200, "Bauxite": 2100,
                         "Limestone": 900, "Copper Ore": 7600}

rows = []
for _, m in mines_df.iterrows():
    remaining_reserve = m["_initial_reserve"]
    # per-mine slow production growth/decline factor over the 9 years
    long_term_drift = rng.uniform(-0.015, 0.02)  # monthly compounding drift

    # per-commodity price random walk, shared trajectory nudged per mine
    price_level = commodity_base_price[m["Commodity"]] * rng.uniform(0.95, 1.05)

    month_index = 0
    for year in years:
        for month in months:
            month_index += 1

            # ---- seasonality (mining slows slightly in monsoon months 6-9) ----
            seasonal = 1.0 - 0.08 * np.sin((month - 6) / 12 * 2 * np.pi) ** 2
            if month in (7, 8):
                seasonal *= 0.93  # monsoon dip

            growth_factor = (1 + long_term_drift) ** month_index
            noise = rng.normal(1.0, 0.06)

            production = max(500.0, m["_base_production"] * growth_factor * seasonal * noise)

            # Reserves deplete with extraction (never randomly jump up)
            remaining_reserve = max(0.0, remaining_reserve - production)

            # Workforce loosely tracks production scale with small variation
            workers = max(20, int(m["_base_workers"] * (0.9 + 0.2 * (production / m["_base_production"])) * rng.normal(1, 0.03)))

            # Working hours: ~ standard 26 working days/month, slight variation, ties to workers
            working_hours = max(1000.0, workers * rng.uniform(180, 220))

            productivity_index = production / working_hours

            # Energy consumption scales with production + mine type overhead
            type_overhead = {"Open Cast": 1.0, "Underground": 1.35, "Mixed": 1.15}[m["Mine_Type"]]
            energy_consumption = production * rng.uniform(0.8, 1.3) * type_overhead

            # Commodity price random walk
            price_level = max(200.0, price_level * rng.normal(1.0, 0.02))
            commodity_price = price_level

            revenue = production * commodity_price / 100.0  # scaled to keep numbers reasonable
            operating_cost = revenue * rng.uniform(0.55, 0.85) + workers * rng.uniform(8, 15)
            profit = revenue - operating_cost
            profit_margin = (profit / revenue * 100.0) if revenue > 0 else 0.0

            # Waste generation scales with production + some mine-type effect
            waste_generation = production * rng.uniform(0.15, 0.45) * type_overhead

            # Environmental impact index: composite of waste + energy + randomness
            environmental_impact = (0.5 * waste_generation / 1000 + 0.3 * energy_consumption / 1000
                                     + rng.uniform(0, 8))
            environmental_breach = 1 if (environmental_impact > np.percentile([environmental_impact], 0) and
                                          rng.random() < 0.06) else 0

            # Inspections & violations
            inspections = rng.poisson(1.4) 
            base_violation_rate = {"increasing": 0.35, "decreasing": 0.35,
                                    "stable": 0.25, "spiky": 0.3}[m["_safety_trend"]]
            violations = rng.poisson(base_violation_rate * (1 + inspections * 0.15))
            violation_category = rng.choice(VIOLATION_CATEGORIES) if violations > 0 else None
            violation_status = rng.choice(["Open", "Closed", "Under Review"]) if violations > 0 else None

            # Safety incidents - depends on mine's safety trend personality
            progress = month_index / (len(years) * 12)
            if m["_safety_trend"] == "increasing":
                trend_mult = 0.6 + 0.9 * progress
            elif m["_safety_trend"] == "decreasing":
                trend_mult = 1.5 - 0.9 * progress
            elif m["_safety_trend"] == "spiky":
                trend_mult = 1.0 + (0.8 if rng.random() < 0.06 else 0.0)
            else:
                trend_mult = 1.0

            safety_incidents = rng.poisson(max(0.05, m["_base_incident_rate"] * trend_mult * 0.35))
            incident_category = rng.choice(INCIDENT_CATEGORIES) if safety_incidents > 0 else None
            severity = rng.choice(SEVERITIES, p=SEVERITY_P) if safety_incidents > 0 else None

            risk_factor = round(min(10.0, 0.4 * safety_incidents + 0.3 * violations
                                     + rng.uniform(0, 2.5)), 2)

            # Corrective actions relate to violations
            corrective_actions = violations + rng.poisson(0.2)
            completed_actions = int(corrective_actions * rng.uniform(0.4, 0.9))
            pending_actions = max(0, corrective_actions - completed_actions)
            action_status = rng.choice(ACTION_STATUS) if corrective_actions > 0 else "Completed"

            rows.append({
                "Mine_ID": m["Mine_ID"],
                "Mine_Name": m["Mine_Name"],
                "Region": m["Region"],
                "State": m["State"],
                "Mine_Type": m["Mine_Type"],
                "Commodity": m["Commodity"],
                "Year": year,
                "Month": month,
                "Date": pd.Timestamp(year=year, month=month, day=1),
                "Production": round(production, 2),
                "Initial_Reserve": round(m["_initial_reserve"], 2),
                "Remaining_Reserve": round(remaining_reserve, 2),
                "Workers": workers,
                "Working_Hours": round(working_hours, 1),
                "Productivity_Index": round(productivity_index, 4),
                "Energy_Consumption": round(energy_consumption, 2),
                "Commodity_Price": round(commodity_price, 2),
                "Revenue": round(revenue, 2),
                "Operating_Cost": round(operating_cost, 2),
                "Profit": round(profit, 2),
                "Profit_Margin": round(profit_margin, 2),
                "Waste_Generation": round(waste_generation, 2),
                "Environmental_Impact": round(environmental_impact, 3),
                "Environmental_Breach": environmental_breach,
                "Inspections": int(inspections),
                "Violations": int(violations),
                "Violation_Category": violation_category,
                "Violation_Status": violation_status,
                "Safety_Incidents": int(safety_incidents),
                "Incident_Category": incident_category,
                "Severity": severity,
                "Risk_Factor": risk_factor,
                "Corrective_Actions": int(corrective_actions),
                "Completed_Actions": int(completed_actions),
                "Pending_Actions": int(pending_actions),
                "Action_Status": action_status,
            })

df = pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3. INTENTIONAL DATA QUALITY ISSUES
# ------------------------------------------------------------------

# --- 3a. Inconsistent category labels (Mine_Type + Violation_Category) ---
dirty_idx = rng.choice(df.index, size=int(0.05 * len(df)), replace=False)
for idx in dirty_idx:
    correct = df.loc[idx, "Mine_Type"]
    variants = MINE_TYPE_VARIANTS.get(correct)
    if variants:
        df.loc[idx, "Mine_Type"] = rng.choice(variants)

dirty_idx2 = rng.choice(df[df["Violation_Category"].notna()].index,
                         size=min(120, df["Violation_Category"].notna().sum()), replace=False)
for idx in dirty_idx2:
    val = df.loc[idx, "Violation_Category"]
    case_variant = rng.choice([val.upper(), val.lower(), val])
    df.loc[idx, "Violation_Category"] = case_variant

# --- 3b. Missing values (~0.2-0.5% in selected columns) ---
cols_for_missing = ["Working_Hours", "Energy_Consumption", "Commodity_Price", "Environmental_Impact"]
for col in cols_for_missing:
    frac = rng.uniform(0.002, 0.005)
    n_missing = int(len(df) * frac)
    miss_idx = rng.choice(df.index, size=n_missing, replace=False)
    df.loc[miss_idx, col] = np.nan

# --- 3c. A few realistic outliers ---
outlier_cols = ["Production", "Energy_Consumption", "Waste_Generation", "Safety_Incidents"]
for col in outlier_cols:
    n_outliers = rng.integers(3, 7)
    out_idx = rng.choice(df.index, size=n_outliers, replace=False)
    multiplier = rng.uniform(3.5, 6.0)
    if col == "Safety_Incidents":
        df.loc[out_idx, col] = (df.loc[out_idx, col].astype(float) * multiplier + 5).astype(int)
    else:
        df.loc[out_idx, col] = df.loc[out_idx, col].astype(float) * multiplier

# --- 3d. Duplicate records (~1%, ~32 rows) ---
n_dupes = 32
dupe_rows = df.sample(n=n_dupes, random_state=SEED)
df = pd.concat([df, dupe_rows], ignore_index=True)

# Shuffle rows slightly (but keep it deterministic) so duplicates aren't all at the end
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Drop helper columns if any leaked in
df = df.loc[:, ~df.columns.str.startswith("_")]

# ------------------------------------------------------------------
# 4. SAVE
# ------------------------------------------------------------------
OUTPUT_FILE = "minewatch_data.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("MineWatch synthetic dataset generated.")
print(f"  Rows:    {len(df):,}")
print(f"  Columns: {df.shape[1]}")
print(f"  Saved to: {OUTPUT_FILE}")
print(f"  Duplicate rows (full-row): {df.duplicated().sum()}")
print(f"  Missing values total: {int(df.isna().sum().sum())}")
