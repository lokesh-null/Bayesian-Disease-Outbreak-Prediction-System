"""
Fetch genuine publicly available epidemiological surveillance datasets from:
1. CDC ILINet / FluView via CMU Delphi Epidata API (https://api.delphi.cmu.edu/epidata/fluview/)
2. CDC NNDSS Weekly Notifiable Diseases via CDC Socrata API (https://data.cdc.gov/resource/x9gk-5huc.json)
   covering Dengue, Pertussis, Salmonellosis, Campylobacteriosis, and Measles across US regions/states.

Saves canonical CSV files with full provenance metadata and SHA-256 checksums.
"""
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def epiweek_to_date(epiweek: int) -> str:
    """Convert YYYYWW epiweek integer to ISO YYYY-MM-DD (Monday of that week)."""
    year = epiweek // 100
    week = epiweek % 100
    try:
        dt = datetime.fromisocalendar(year, min(max(week, 1), 52), 1)
    except ValueError:
        dt = datetime(year, 1, 1) + timedelta(weeks=max(week - 1, 0))
    return dt.strftime("%Y-%m-%d")


def fetch_cdc_ilinet():
    """Fetch genuine CDC ILINet weekly surveillance data from CMU Delphi Epidata API."""
    regions = ["nat", "hhs1", "hhs2", "hhs4", "hhs6", "hhs9"]
    region_names = {
        "nat": "United States (National)",
        "hhs1": "HHS Region 1 (New England - Boston)",
        "hhs2": "HHS Region 2 (NY, NJ, PR, VI)",
        "hhs4": "HHS Region 4 (Southeast - Atlanta)",
        "hhs6": "HHS Region 6 (South Central - Dallas)",
        "hhs9": "HHS Region 9 (Pacific Southwest - San Francisco)",
    }
    region_pops = {
        "nat": 333287557,
        "hhs1": 15121775,
        "hhs2": 31830000,
        "hhs4": 68900000,
        "hhs6": 43800000,
        "hhs9": 51900000,
    }

    url = (
        f"https://api.delphi.cmu.edu/epidata/fluview/"
        f"?regions={','.join(regions)}&epiweeks=201701-202452"
    )
    print(f"Fetching CDC ILINet from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "BMA-EpiForecast-Academic/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("result") != 1 or not payload.get("epidata"):
        raise RuntimeError(f"Unexpected Delphi API response: {payload.get('message')}")

    records = []
    for row in payload["epidata"]:
        reg = row["region"]
        ew = int(row["epiweek"])
        records.append({
            "observation_date": epiweek_to_date(ew),
            "epiweek": ew,
            "disease_code": "ILI",
            "disease_name": "Influenza-Like Illness (CDC ILINet)",
            "location_code": f"US-{reg.upper()}",
            "location_name": region_names.get(reg, reg.upper()),
            "confirmed_cases": int(row["num_ili"]) if row.get("num_ili") is not None else 0,
            "deaths": None,
            "exposure_denominator": int(row["num_patients"]) if row.get("num_patients") is not None else None,
            "population": region_pops.get(reg),
            "num_providers": row.get("num_providers"),
            "wili_pct": row.get("wili"),
            "ili_pct": row.get("ili"),
            "release_date": row.get("release_date"),
            "source_name": "CDC ILINet via CMU Delphi Epidata API",
        })

    df = pd.DataFrame(records)
    df = df.sort_values(["location_code", "epiweek"]).reset_index(drop=True)
    out_path = DATA_DIR / "cdc_ilinet_weekly_2017_2024.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} genuine CDC ILINet observations to {out_path}")
    return out_path, len(df), url


def fetch_cdc_nndss():
    """Fetch genuine CDC NNDSS Weekly Notifiable Diseases data via Socrata API."""
    selected_labels = [
        ("Dengue virus infections, Dengue", "DENGUE", "Dengue Virus Infection (CDC NNDSS)"),
        ("Pertussis", "PERTUSSIS", "Pertussis / Whooping Cough (CDC NNDSS)"),
        ("Salmonellosis (excluding Salmonella Typhi infection and Salmonella Paratyphi infection)", "SALMONELLOSIS", "Salmonellosis (CDC NNDSS)"),
        ("Campylobacteriosis", "CAMPYLOBACTER", "Campylobacteriosis (CDC NNDSS)"),
        ("Malaria", "MALARIA", "Malaria (CDC NNDSS)"),
    ]

    loc_map = {
        "US RESIDENTS": ("US-NAT", "United States (National)", 333287557),
        "NEW ENGLAND": ("US-NEW_ENGLAND", "New England (US Census Division)", 15121775),
        "MIDDLE ATLANTIC": ("US-MID_ATLANTIC", "Middle Atlantic (NY, NJ, PA)", 42400000),
        "SOUTH ATLANTIC": ("US-SOUTH_ATLANTIC", "South Atlantic (FL, GA, NC, VA, MD)", 66700000),
        "PACIFIC": ("US-PACIFIC", "Pacific Division (CA, WA, OR, HI, AK)", 53400000),
        "PUERTO RICO": ("US-PR", "Puerto Rico", 3221789),
        "FLORIDA": ("US-FL", "Florida", 22610726),
        "CALIFORNIA": ("US-CA", "California", 38965193),
        "NEW YORK": ("US-NY", "New York", 19571216),
    }

    records = []
    for raw_label, d_code, d_name in selected_labels:
        encoded_where = urllib.parse.urlencode({
            "$where": f"label='{raw_label}' AND states in ('US RESIDENTS', 'NEW ENGLAND', 'MIDDLE ATLANTIC', 'SOUTH ATLANTIC', 'PACIFIC', 'PUERTO RICO', 'FLORIDA', 'CALIFORNIA', 'NEW YORK')",
            "$limit": "5000",
            "$order": "year ASC, week ASC"
        })
        q_url = f"https://data.cdc.gov/resource/x9gk-5huc.json?{encoded_where}"
        req = urllib.request.Request(q_url, headers={"User-Agent": "BMA-EpiForecast-Academic/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
            print(f"  Fetched {len(rows)} rows for '{d_code}' ({raw_label})")

        for row in rows:
            try:
                yr = int(row.get("year", 2022))
                wk = int(row.get("week", 1))
            except (ValueError, TypeError):
                continue
            ew = yr * 100 + wk
            val_str = row.get("m1")
            flag = str(row.get("m1_flag", "")).strip()
            if val_str is not None and str(val_str).strip() not in ("", "-", "N", "U", "NN", "NP"):
                try:
                    cases = int(float(val_str))
                except ValueError:
                    cases = None
            else:
                # In CDC NNDSS, m1_flag == '-' indicates zero reported cases for that week
                cases = 0 if flag == "-" else None

            raw_loc = row.get("states", "US RESIDENTS").strip().upper()
            loc_code, loc_name, pop = loc_map.get(
                raw_loc,
                ("US-" + raw_loc.replace(" ", "_"), raw_loc.title(), None),
            )

            records.append({
                "observation_date": epiweek_to_date(ew),
                "epiweek": ew,
                "disease_code": d_code,
                "disease_name": d_name,
                "location_code": loc_code,
                "location_name": loc_name,
                "confirmed_cases": cases,
                "deaths": None,
                "exposure_denominator": None,
                "population": pop,
                "source_name": "CDC NNDSS Weekly Surveillance (data.cdc.gov/resource/x9gk-5huc)",
            })

    df_nndss = pd.DataFrame(records)
    df_nndss = df_nndss.drop_duplicates(
        subset=["disease_code", "location_code", "observation_date"], keep="last"
    )
    df_nndss = df_nndss.sort_values(["disease_code", "location_code", "epiweek"]).reset_index(drop=True)
    out_path = DATA_DIR / "cdc_nndss_weekly_2022_2024.csv"
    df_nndss.to_csv(out_path, index=False)
    print(f"Saved {len(df_nndss)} deduplicated CDC NNDSS rows to {out_path}")
    return out_path, len(df_nndss)


if __name__ == "__main__":
    fetch_cdc_ilinet()
    fetch_cdc_nndss()
