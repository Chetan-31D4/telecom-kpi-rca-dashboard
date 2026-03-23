"""
Generalized RCA engine extracted from phase_6 notebook.
Works for any KPI (CSSR, PSR, LUSR, etc.) based on uploaded files.
"""

import pandas as pd
import numpy as np
import datetime
import re


# ── File loading & normalization ──────────────────────────────────────────────

def load_datewise_kpi(df_raw: pd.DataFrame, date_col: str, kpi_col: str,
                      dayfirst: bool = True) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    df = df[["date", kpi_col]].copy()
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_hourly_kpi(df_raw: pd.DataFrame, date_col: str, hour_col: str,
                    kpi_col: str, dayfirst: bool = True) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df[date_col], dayfirst=dayfirst,
                                errors="coerce").dt.normalize()
    df["hour"] = (
        df[hour_col].astype(str)
        .str.extract(r"(\d{1,2})")[0]
        .astype(float).astype("Int64")
    )
    df = df[["date", "hour", kpi_col]].copy()
    df = df.dropna(subset=["date", "hour"])
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    return df


def normalize_cc_file(df_raw: pd.DataFrame, date_col: str = "D1DATE",
                      hour_col: str = "D1HOUR", msc_col: str = "MSC NAME",
                      ccid_col: str = "CC ID",
                      value_col: str = "INTERNAL CLEAR CODES",
                      dayfirst: bool = True) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df[date_col], dayfirst=dayfirst,
                                errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # Try parsing hour as time first, fall back to regex extract
    hour_parsed = pd.to_datetime(df[hour_col].astype(str), errors="coerce").dt.hour
    if hour_parsed.notna().sum() > 0:
        df["hour"] = hour_parsed
    else:
        df["hour"] = (
            df[hour_col].astype(str)
            .str.extract(r"(\d{1,2})")[0]
            .astype(float).astype("Int64")
        )

    df["msc"] = df[msc_col].astype(str).str.upper().str.strip()
    df["cc_id"] = df[ccid_col].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    df = df[["date", "hour", "msc", "cc_id", "value"]]
    df = df.dropna(subset=["date", "hour", "msc", "cc_id"])
    return df


# ── Dip detection ─────────────────────────────────────────────────────────────

def add_prev_and_delta(df: pd.DataFrame, kpi_col: str) -> pd.DataFrame:
    df = df.copy()
    df["prev"] = df[kpi_col].shift(1)
    df["delta"] = df[kpi_col] - df["prev"]
    return df


def add_prev_and_delta_hourly(df: pd.DataFrame, kpi_col: str) -> pd.DataFrame:
    df = df.copy()
    df["prev"] = df.groupby("date")[kpi_col].shift(1)
    df["delta"] = df[kpi_col] - df["prev"]
    return df


def get_top_dips(df: pd.DataFrame, kpi_col: str,
                 top_n: int = 5) -> pd.DataFrame:
    dips = df[df["delta"] < 0].copy()
    dips = dips.sort_values("delta")
    return dips.head(top_n)[["date", kpi_col, "prev", "delta"]].reset_index(drop=True)


def get_worst_hours_per_dip(hour_df: pd.DataFrame,
                            dip_dates_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    hour_df = hour_df.copy()
    hour_df["date"] = pd.to_datetime(hour_df["date"], errors="coerce").dt.normalize()
    dip_dates_df = dip_dates_df.copy()
    dip_dates_df["date"] = pd.to_datetime(dip_dates_df["date"],
                                          errors="coerce").dt.normalize()

    for dip_date in dip_dates_df["date"]:
        day_df = hour_df[
            (hour_df["date"] == dip_date) & (hour_df["delta"].notna())
        ]
        if day_df.empty:
            continue
        worst_row = day_df.loc[day_df["delta"].idxmin()]
        results.append({
            "date": dip_date,
            "worst_hour": int(worst_row["hour"]),
            "delta": worst_row["delta"]
        })
    return pd.DataFrame(results)


# ── CC-share RCA ──────────────────────────────────────────────────────────────

def run_share_rca(worst_hour_df: pd.DataFrame, cc_df: pd.DataFrame,
                  target_cc_ids: list[str], kpi_name: str,
                  ref_days: int = 7, fallback_days: int | None = 14):
    results = []
    skip_logs = []

    for _, r in worst_hour_df.iterrows():
        event_date = r["date"]
        event_hour = int(r["worst_hour"])

        ref_date_1 = event_date - pd.Timedelta(days=ref_days)
        ref_date_2 = (event_date - pd.Timedelta(days=fallback_days)
                      if fallback_days else None)

        # Find reference
        ref_rows_1 = cc_df[
            (cc_df["date"] == ref_date_1) & (cc_df["hour"] == event_hour)
        ]
        ref_rows_2 = (cc_df[
            (cc_df["date"] == ref_date_2) & (cc_df["hour"] == event_hour)
        ] if ref_date_2 is not None else pd.DataFrame())

        if not ref_rows_1.empty:
            ref_date = ref_date_1
        elif fallback_days and not ref_rows_2.empty:
            ref_date = ref_date_2
        else:
            skip_logs.append({
                "date": event_date, "hour": event_hour,
                "reason": f"No reference at -{ref_days}d" +
                          (f" or -{fallback_days}d" if fallback_days else "")
            })
            continue

        # Current data
        curr_all = cc_df[
            (cc_df["date"] == event_date) & (cc_df["hour"] == event_hour)
        ]
        if curr_all.empty:
            skip_logs.append({
                "date": event_date, "hour": event_hour,
                "reason": "No CC data for current date/hour"
            })
            continue

        # MSC loop
        for msc in curr_all["msc"].unique():
            curr = curr_all[curr_all["msc"] == msc]
            ref = cc_df[
                (cc_df["date"] == ref_date) &
                (cc_df["hour"] == event_hour) &
                (cc_df["msc"] == msc)
            ]
            if curr.empty or ref.empty:
                skip_logs.append({
                    "date": event_date, "hour": event_hour, "msc": msc,
                    "reason": "MSC missing curr/ref"
                })
                continue

            total_curr = curr["value"].sum()
            total_ref = ref["value"].sum()
            if total_curr == 0 or total_ref == 0:
                skip_logs.append({
                    "date": event_date, "hour": event_hour, "msc": msc,
                    "reason": "Zero total value"
                })
                continue

            target_ids_upper = [t.upper().strip() for t in target_cc_ids]
            curr_cc = curr[curr["cc_id"].isin(target_ids_upper)]
            ref_cc = ref[ref["cc_id"].isin(target_ids_upper)]

            common_cc = set(curr_cc["cc_id"]) & set(ref_cc["cc_id"])
            if not common_cc:
                skip_logs.append({
                    "date": event_date, "hour": event_hour, "msc": msc,
                    "reason": "No common CC_ID"
                })
                continue

            for ccid in common_cc:
                cc_curr_val = curr_cc[curr_cc["cc_id"] == ccid]["value"].sum()
                cc_ref_val = ref_cc[ref_cc["cc_id"] == ccid]["value"].sum()
                share_curr = cc_curr_val / total_curr
                share_ref = cc_ref_val / total_ref
                share_delta = share_curr - share_ref

                results.append({
                    "KPI": kpi_name,
                    "Date": event_date,
                    "Hour": event_hour,
                    "MSC": msc,
                    "CC_ID": ccid,
                    "Ref_Date": ref_date,
                    "Total_Current": round(total_curr, 2),
                    "Total_Reference": round(total_ref, 2),
                    "CC_Current": round(cc_curr_val, 2),
                    "CC_Reference": round(cc_ref_val, 2),
                    "Share_Current": round(share_curr, 6),
                    "Share_Reference": round(share_ref, 6),
                    "Share_Delta": round(share_delta, 6),
                    "Direction": "Increased" if share_delta > 0 else "Decreased"
                })

    rca_df = pd.DataFrame(results)
    skip_df = pd.DataFrame(skip_logs)

    if not rca_df.empty:
        rca_df["Impact_Rank"] = (
            rca_df.groupby(["Date", "Hour", "MSC"])["Share_Delta"]
            .transform(lambda x: x.abs().rank(method="dense", ascending=False))
            .astype(int)
        )
        rca_df = rca_df.sort_values(
            by=["Date", "Hour", "MSC", "Impact_Rank"]
        ).reset_index(drop=True)

    return rca_df, skip_df


# ── LLM summary helpers ──────────────────────────────────────────────────────

CC_ACTION_MAP = {
    "A03": "INTERWORKING FAILED",
    "A06": "CHANNEL UNACCEPTABLE",
    "A07": "CALL AWARDED AND BEING DELIVERED IN AN ESTABLISHED CHANNEL",
    "A09": "PRE-EMPTION",
    "A0A": "PRE-EMPTION - CIRCUIT RESERVED FOR REUSE",
    "A11": "NORMAL CALL CLEARING",
    "706": "UNALLOCATED (UNASSIGNED) NUMBER",
    "603": "DESTINATION OUT OF ORDER",
    "B04": "NO USER RESPONDING",
    "B05": "USER ALERTING, NO ANSWER",
    "B13": "RESOURCES UNAVAILABLE, UNSPECIFIED",
    "B16": "NORMAL, UNSPECIFIED",
    "B1A": "REQUESTED CIRCUIT/CHANNEL NOT AVAILABLE",
    "B1C": "NETWORK OUT OF ORDER",
    "B1B": "QUALITY OF SERVICE UNAVAILABLE",
    "B2C": "SERVICE OR OPTION NOT AVAILABLE",
    "B2D": "SERVICE OR OPTION NOT IMPLEMENTED",
    "B17": "USER BUSY",
    "B32": "INTERWORKING, UNSPECIFIED",
    "B1E": "TEMPORARY FAILURE",
    "811": "PROTOCOL ERROR, UNSPECIFIED",
    "817": "TIMER EXPIRY",
    "10":  "Check radio access failures and retry mechanisms",
    "12":  "Analyze abnormal call releases from core network",
    "15":  "Validate inter-MSC handover stability",
    "0307": "LOCATION UPDATE REJECT - ILLEGAL MS",
    "0811": "GPRS DETACH - REATTACH REQUIRED",
    "0812": "GPRS DETACH - REATTACH NOT REQUIRED",
    "081B": "ROUTING AREA UPDATE REJECT - NO SUITABLE CELLS IN LA",
    "081C": "SERVICE REJECT - CONGESTION",
    "0B13": "ATTACH REJECT - ROAMING NOT ALLOWED IN THIS LA",
}


def build_dip_groups(rca_df: pd.DataFrame, top_n: int = 3):
    if rca_df.empty:
        return []

    top_cc = (
        rca_df.sort_values("Impact_Rank")
        .groupby(["Date", "Hour", "MSC"]).head(top_n)
    )

    groups = []
    for (date, hour, msc), g in top_cc.groupby(["Date", "Hour", "MSC"]):
        cc_details = []
        for _, row in g.iterrows():
            cc_details.append({
                "cc_id": row["CC_ID"],
                "definition": CC_ACTION_MAP.get(row["CC_ID"],
                                                "Definition not available"),
                "share_delta": round(row["Share_Delta"], 6),
                "impact_rank": int(row["Impact_Rank"])
            })
        groups.append({
            "date": str(date)[:10] if hasattr(date, 'strftime') else str(date),
            "hour": hour,
            "msc": msc,
            "cc_details": cc_details
        })
    return groups


# ── LLM RCA generation ───────────────────────────────────────────────────────

RCA_SYSTEM_PROMPT = (
    "You are a senior telecom core-network RCA engineer writing incident "
    "root cause analyses for operations teams. Be concise, precise, and "
    "use telecom terminology (signaling, interworking, routing, congestion). "
    "Use ONLY the given clear code definitions — never invent meanings."
)

RCA_USER_TEMPLATE = """KPI impacted: {kpi}
Date: {date}  |  Busy Hour: {hour}  |  MSC: {msc}

Top contributing clear codes:
{cc_block}

Write a 3-5 sentence RCA paragraph explaining:
1. What failed (using the exact CC definitions)
2. The likely network-level root cause
3. Recommended next steps for the NOC team"""


def _format_cc_block(cc_details: list[dict]) -> str:
    lines = []
    for c in cc_details:
        meaning = c["definition"]
        if meaning == "Definition not available":
            meaning = "Unknown failure category"
        lines.append(
            f"- {c['cc_id']}: {meaning}  "
            f"(share change = {c['share_delta']:+.6f}, rank {c['impact_rank']})"
        )
    return "\n".join(lines)


def generate_llm_rca(dip_groups: list[dict], kpi_name: str,
                     api_key: str, model: str = "gpt-4o-mini") -> list[dict]:
    """Generate natural-language RCA for each dip group via OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    results = []

    for dip in dip_groups:
        cc_block = _format_cc_block(dip["cc_details"])
        user_msg = RCA_USER_TEMPLATE.format(
            kpi=kpi_name, date=dip["date"],
            hour=dip["hour"], msc=dip["msc"],
            cc_block=cc_block,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RCA_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        summary = response.choices[0].message.content.strip()

        results.append({
            "date": dip["date"],
            "hour": dip["hour"],
            "msc": dip["msc"],
            "top_ccs": ", ".join(c["cc_id"] for c in dip["cc_details"]),
            "summary": summary,
        })

    return results
