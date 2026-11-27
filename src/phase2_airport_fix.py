"""
Phase 2 — October Airport Code Fix

Goal: replace the numeric airport codes used in October with three-letter IATA codes,
without losing any row. Two independent methods are used and compared:
    (A) matching October schedules with September/November schedules
    (B) the official BTS airport code tables (downloaded, needs Internet ON)

Runs on Kaggle with two inputs attached (Add Input):
    - Phase 0 notebook output   -> flights.parquet
    - usdot/flight-delays       -> airports.csv
Outputs (in /kaggle/working):
    flights_fixed.parquet   -> input for Phase 3 (not pushed to GitHub)
    airport_id_map.csv      -> the 307 codes with their IATA code and source
"""

# %% 2.1 — Load
import os
import pandas as pd

parquet_path = None
raw_dir = None
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "flights.parquet" in files:
        parquet_path = folder + "/flights.parquet"
    if "airports.csv" in files:
        raw_dir = folder

df = pd.read_parquet(parquet_path)
airports = pd.read_csv(raw_dir + "/airports.csv")

rows_before = len(df)
print("Flights:", rows_before)
print("Airports in airports.csv:", len(airports))

# %% 2.1 — Codes used in October
october = df[df["MONTH"] == 10]

october_codes = set(october["ORIGIN_AIRPORT"]) | set(october["DESTINATION_AIRPORT"])
print("Numeric codes to fix:", len(october_codes))

# %% 2.2 — Method A: schedules from September and November
# A schedule is identified by these 6 columns (same weekly flight)
key = ["AIRLINE", "FLIGHT_NUMBER", "DAY_OF_WEEK",
       "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE"]

sep_nov = df[df["MONTH"].isin([9, 11])]
reference = sep_nov[key + ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]].drop_duplicates()
reference = reference.rename(columns={"ORIGIN_AIRPORT": "ORIGIN_IATA",
                                      "DESTINATION_AIRPORT": "DEST_IATA"})

print("Schedules in September + November:", len(reference))

# %% 2.2 — Method A: schedules from October
october_small = october[key + ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]].drop_duplicates()
october_small = october_small.rename(columns={"ORIGIN_AIRPORT": "ORIGIN_ID",
                                              "DESTINATION_AIRPORT": "DEST_ID"})

print("Schedules in October:", len(october_small))

# %% 2.2 — Method A: match October schedules with September/November schedules
matched = october_small.merge(reference, on=key, how="inner")

print("Matched schedules:", len(matched))
print(matched[["ORIGIN_ID", "ORIGIN_IATA", "DEST_ID", "DEST_IATA"]].head(10))

# %% 2.3 — Method A: collect (numeric code, IATA code) pairs
origin_pairs = matched[["ORIGIN_ID", "ORIGIN_IATA"]]
origin_pairs = origin_pairs.rename(columns={"ORIGIN_ID": "ID", "ORIGIN_IATA": "IATA"})

dest_pairs = matched[["DEST_ID", "DEST_IATA"]]
dest_pairs = dest_pairs.rename(columns={"DEST_ID": "ID", "DEST_IATA": "IATA"})

mapping_a = pd.concat([origin_pairs, dest_pairs]).drop_duplicates()

print("Different (ID, IATA) pairs:", len(mapping_a))
print("Different IDs:", mapping_a["ID"].nunique())
print(mapping_a.head())

# %% 2.3 — Method A: checks
print("IDs matched to more than one IATA code:", mapping_a["ID"].duplicated().sum())
print("IATA codes matched to more than one ID:", mapping_a["IATA"].duplicated().sum())

not_found_a = october_codes - set(mapping_a["ID"])
print("\nIDs not found by method A:", len(not_found_a))

affected = october[october["ORIGIN_AIRPORT"].isin(not_found_a) | october["DESTINATION_AIRPORT"].isin(not_found_a)]
print("October flights using these IDs:", len(affected), "(", round(len(affected) / len(october) * 100, 2), "% )")

# %% 2.4 — Method B: download the BTS tables (Latin-1 encoded; read everything as text)
url_id = "https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_NVecbeg_VQ"    # L_AIRPORT_ID
url_iata = "https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_NVecbeg"     # L_AIRPORT

table_id = pd.read_csv(url_id, encoding="latin-1", dtype=str)
table_iata = pd.read_csv(url_iata, encoding="latin-1", dtype=str)

print("L_AIRPORT_ID rows:", len(table_id))
print(table_id.head(3))
print("\nL_AIRPORT rows:", len(table_iata))
print(table_iata.head(3))

# %% 2.4 — Method B: join the two tables on the airport name
table_id = table_id.rename(columns={"Code": "ID"})
table_iata = table_iata.rename(columns={"Code": "IATA"})

mapping_b = table_id.merge(table_iata, on="Description", how="inner")
mapping_b = mapping_b[mapping_b["ID"].isin(october_codes)]

print("IDs found:", mapping_b["ID"].nunique(), "of", len(october_codes))
print("Rows:", len(mapping_b))

# %% 2.4 — Method B: codes with more than one candidate -> keep codes used in our data
multi = mapping_b[mapping_b["ID"].duplicated(keep=False)]
print("IDs with more than one candidate:")
print(multi)

iata_list = set(airports["IATA_CODE"])
mapping_b = mapping_b[mapping_b["IATA"].isin(iata_list)]

print("\nAfter keeping only codes in airports.csv:")
print("IDs found:", mapping_b["ID"].nunique(), "| Rows:", len(mapping_b))

# %% 2.5 — Which code did method B lose? (today's BTS table, not 2015)
lost = october_codes - set(mapping_b["ID"])
print("Lost ID:", lost)

lost_row = table_id[table_id["ID"].isin(lost)]
print(lost_row)

lost_name = lost_row["Description"].iloc[0]
print(table_iata[table_iata["Description"] == lost_name])

# %% 2.5 — Compare method A and method B
compare = mapping_a.merge(mapping_b[["ID", "IATA"]], on="ID", how="outer",
                          suffixes=("_A", "_B"))

both = compare.dropna()
print("Found by both methods:", len(both))
print("Same answer:", (both["IATA_A"] == both["IATA_B"]).sum())
print("Different answer:", (both["IATA_A"] != both["IATA_B"]).sum())

print("\nFound only by A:")
print(compare[compare["IATA_B"].isna()])

print("\nFound only by B:", compare["IATA_A"].isna().sum())

# %% 2.6 — Final mapping: method A, otherwise method B
final_map = compare.copy()
final_map["IATA"] = final_map["IATA_A"].fillna(final_map["IATA_B"])

final_map["source"] = "both"
final_map.loc[final_map["IATA_A"].isna(), "source"] = "B only"
final_map.loc[final_map["IATA_B"].isna(), "source"] = "A only"

print(final_map["source"].value_counts())
print("Total:", len(final_map))
print("All codes in airports.csv:", final_map["IATA"].isin(iata_list).all())
print("Same IATA used twice:", final_map["IATA"].duplicated().sum())

# %% 2.6 — Apply the mapping to October
code_dict = dict(zip(final_map["ID"], final_map["IATA"]))
print("Example: 14747 ->", code_dict["14747"])

is_october = df["MONTH"] == 10
df.loc[is_october, "ORIGIN_AIRPORT"] = df.loc[is_october, "ORIGIN_AIRPORT"].map(code_dict)
df.loc[is_october, "DESTINATION_AIRPORT"] = df.loc[is_october, "DESTINATION_AIRPORT"].map(code_dict)

print(df.loc[is_october, ["MONTH", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]].head())

# %% 2.7 — Checks: no row lost, no numeric code left, every code valid
print("Rows before:", rows_before, "| Rows after:", len(df))

print("\nORIGIN_AIRPORT")
print("  missing values    :", df["ORIGIN_AIRPORT"].isna().sum())
print("  numeric codes left:", df["ORIGIN_AIRPORT"].str.isdigit().sum())
print("  not in airports.csv:", (~df["ORIGIN_AIRPORT"].isin(iata_list)).sum())

print("\nDESTINATION_AIRPORT")
print("  missing values    :", df["DESTINATION_AIRPORT"].isna().sum())
print("  numeric codes left:", df["DESTINATION_AIRPORT"].str.isdigit().sum())
print("  not in airports.csv:", (~df["DESTINATION_AIRPORT"].isin(iata_list)).sum())

# %% 2.7 — Save (before the distance check, so a stuck kernel cannot lose the result)
df.to_parquet("/kaggle/working/flights_fixed.parquet", index=False)
final_map[["ID", "IATA", "source"]].to_csv("/kaggle/working/airport_id_map.csv", index=False)

print("Saved:", os.listdir("/kaggle/working"))

# %% 2.7 — Distance check: a route has the same distance in every month
small = df[["MONTH", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE"]].drop_duplicates()

route_oct = small[small["MONTH"] == 10].groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["DISTANCE"].median()
route_other = small[small["MONTH"] != 10].groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["DISTANCE"].median()

routes = pd.DataFrame({"oct": route_oct, "other": route_other}).dropna()
routes["difference"] = (routes["oct"] - routes["other"]).abs()

print("Routes checked:", len(routes))
print("Same distance:", (routes["difference"] == 0).sum())
print("Different distance:", (routes["difference"] > 0).sum())
print("Largest difference (miles):", routes["difference"].max())
