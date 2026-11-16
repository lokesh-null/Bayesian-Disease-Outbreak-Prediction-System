# Data Sources & Provenance Documentation

## 1. Overview of Genuine Epidemiological Datasets

All datasets bundled and analyzed in this system originate from official public health surveillance repositories. No synthetic or fabricated datasets are used for any epidemiological analysis or performance claim.

| Dataset File | Official Repository & API | Diseases Covered | Geographic Regions | Time Span | Total Rows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [cdc_ilinet_weekly_2017_2024.csv](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/data/cdc_ilinet_weekly_2017_2024.csv) | **U.S. CDC Outpatient Influenza-like Illness Surveillance Network (ILINet)** accessed via the **CMU Delphi Epidata API** (`https://api.delphi.cmu.edu/epidata/fluview/`) | Influenza-Like Illness (`ILI` weekly clinical case counts `num_ili`, patient denominator `num_patients`, and reporting providers) | `US-NAT` (National), `US-HHS1` (New England), `US-HHS2` (NY/NJ/PR), `US-HHS4` (Southeast), `US-HHS6` (South Central), `US-HHS9` (Pacific Southwest) | Epiweek `201701` (Jan 2017) – `202452` (Dec 2024) | 2,502 weekly observations |
| [cdc_nndss_weekly_2022_2024.csv](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/data/cdc_nndss_weekly_2022_2024.csv) | **U.S. CDC National Notifiable Diseases Surveillance System (NNDSS)** accessed via the **CDC Socrata Open Data API** (`https://data.cdc.gov/resource/x9gk-5huc.json`) | `DENGUE` (Dengue virus infections), `PERTUSSIS` (Whooping Cough), `SALMONELLOSIS`, `CAMPYLOBACTER` (Campylobacteriosis), `MALARIA` | `US-NAT`, `US-NEW_ENGLAND`, `US-MID_ATLANTIC`, `US-SOUTH_ATLANTIC`, `US-PACIFIC`, `US-PR` (Puerto Rico), `US-FL`, `US-CA`, `US-NY` | Epiweek `202201` (Jan 2022) – `202452` (Dec 2024) | 7,020 weekly observations |

---

## 2. Attribution, Licensing & Download Provenance

1. **CDC ILINet / FluView via CMU Delphi Group**:
   - **Original Source**: Centers for Disease Control and Prevention (CDC), National Center for Immunization and Respiratory Diseases (NCIRD).
   - **API Distribution**: Carnegie Mellon University Delphi Research Group (`https://cmu-delphi.github.io/delphi-epidata/`).
   - **License**: Public Domain (U.S. Government Work under 17 U.S.C. § 105).
   - **Download Script**: [fetch_real_datasets.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/scripts/fetch_real_datasets.py).

2. **CDC NNDSS Weekly Provisional Surveillance Tables**:
   - **Original Source**: Centers for Disease Control and Prevention (CDC), Division of Health Informatics and Surveillance (DHIS).
   - **Portal URL**: `https://data.cdc.gov/NNDSS/NNDSS-Weekly-Data/x9gk-5huc`
   - **License**: Public Domain (U.S. Government Work).
   - **Download Script**: [fetch_real_datasets.py](file:///c:/Users/lokes/Downloads/ML%20presentation/backend/scripts/fetch_real_datasets.py).

---

## 3. Re-Fetching or Updating Live Surveillance Data

To refresh the local CSV files directly from the live CMU Delphi and CDC Socrata APIs:
```bash
python backend/scripts/fetch_real_datasets.py
```
If external network access is unavailable, the system automatically operates from the pre-bundled canonical CSV datasets in `backend/data/` and supports interactive CSV/Excel file uploads via the **Data Explorer** page (`POST /api/v1/datasets/upload`).

