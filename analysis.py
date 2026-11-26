import pandas as pd, numpy as np
from scipy.stats import chi2_contingency

# 1. LOAD
contacts = pd.read_csv("contacts_raw.csv")
campaigns = pd.read_csv("campaigns.csv")
events = pd.read_csv("outreach_events.csv")
print("RAW rows:", len(contacts), len(events))

# 2. DATA QUALITY BEFORE CLEANING (the 'scorecard' baseline)
raw_dupes = contacts.duplicated("contact_id").mean()
raw_missing_title = contacts["job_title"].isna().mean() + contacts["job_title"].astype(str).str.strip().isin(["", "N/A"]).mean()
print(f"Duplicate contact rows: {raw_dupes:.1%}")
print(f"Missing/placeholder titles: {raw_missing_title:.1%}")
print(f"Missing employee_count: {contacts['employee_count'].isna().mean():.1%}")

# 3. CLEANING
contacts = contacts.drop_duplicates("contact_id").copy()
contacts["provider"] = contacts["provider"].str.strip().str.title()
contacts["industry"] = contacts["industry"].str.strip()
contacts["job_title"] = contacts["job_title"].astype(str).str.strip().replace({"": np.nan, "N/A": np.nan, "nan": np.nan})
contacts["job_title"] = contacts["job_title"].fillna("Unknown")
contacts["last_verified"] = pd.to_datetime(contacts["last_verified"], format="mixed", dayfirst=False, errors="coerce")
contacts["employee_count"] = contacts["employee_count"].fillna(contacts.groupby("industry")["employee_count"].transform("median"))
events = events.drop_duplicates("event_id").copy()
events["status"] = events["status"].str.strip().str.lower()
events["sent_date"] = pd.to_datetime(events["sent_date"])
df = (events.merge(campaigns[["campaign_id", "segment", "personalization"]], on="campaign_id")
            .merge(contacts[["contact_id", "provider", "email_status", "title_match"]], on="contact_id", how="left"))
print("CLEAN rows:", len(contacts), len(events), "| unmatched events:", df["provider"].isna().sum())

# 4. KPIs
df["delivered"] = (df["status"] == "delivered").astype(int)
df["bounced"] = 1 - df["delivered"]
def kpis(g):
    sent, deliv = len(g), g["delivered"].sum()
    return pd.Series({"sent": sent,
                      "bounce_rate": g["bounced"].mean(),
                      "reply_rate": g["replied"].sum() / deliv,
                      "meeting_rate": g["meeting_booked"].sum() / deliv})
overall = kpis(df)
print("\nOVERALL\n", overall.round(4).to_string())
for col in ["personalization", "provider", "email_status", "title_match", "segment"]:
    print(f"\nBY {col}\n", df.groupby(col).apply(kpis, include_groups=False).round(4).to_string())

# 5. CONTACT DATA QUALITY SCORECARD (per provider)
score = contacts.groupby("provider").agg(
    contacts=("contact_id", "count"),
    valid_email_pct=("email_status", lambda s: (s == "valid").mean()),
    title_match_pct=("title_match", lambda s: (s == "Yes").mean()),
    title_known_pct=("job_title", lambda s: (s != "Unknown").mean()))
score["quality_index"] = (0.5 * score["valid_email_pct"] + 0.5 * score["title_match_pct"]) * 100
print("\nSCORECARD\n", score.round(3).to_string())

# 6. STATISTICS: does personalization change reply rate? (chi-square) + simple CI
d = df[df["delivered"] == 1]
tab = pd.crosstab(d["personalization"], d["replied"])
chi2, p, _, _ = chi2_contingency(tab)
print("\nChi-square personalization vs reply: chi2=%.2f p=%.4f" % (chi2, p))
rate = d.groupby("personalization")["replied"].mean()
n = d.groupby("personalization")["replied"].size()
diff = rate["Personalized"] - rate["Generic"]
se = np.sqrt(rate["Personalized"]*(1-rate["Personalized"])/n["Personalized"] + rate["Generic"]*(1-rate["Generic"])/n["Generic"])
print("Reply-rate lift: %.3f (95%% CI %.3f to %.3f)" % (diff, diff-1.96*se, diff+1.96*se))
tab2 = pd.crosstab(d["title_match"], d["replied"])
chi2b, pb, _, _ = chi2_contingency(tab2)
print("Chi-square title_match vs reply: chi2=%.2f p=%.4f" % (chi2b, pb))
print("Corr(valid email flag, bounce):", round(np.corrcoef((df['email_status']=='valid').astype(int), df['bounced'])[0,1], 3))

# 7. SAVE CLEAN FILES
contacts.to_csv("contacts_clean.csv", index=False)
events.to_csv("events_clean.csv", index=False)
df.to_csv("funnel_model.csv", index=False)
