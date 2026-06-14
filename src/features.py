import math
from typing import List

import pandas as pd

from classes import MatchSession, RANK_STAT_AVERAGES, tier_to_rank_name


def matches_to_dataframe(matches: List[MatchSession]) -> pd.DataFrame:
    rows = []
    for m in matches:
        ke = m.kill_event
        k_loc = ke.player_locations_on_kill[0]["location"]
        v_loc = ke.player_locations_on_kill[1]["location"]
        kill_distance = math.hypot(k_loc["x"] - v_loc["x"], k_loc["y"] - v_loc["y"])

        rows.append({

            "puuid": m.killer.puuid,
            "tier": m.killer.competitive_tier,
            "rank_name": tier_to_rank_name(m.killer.competitive_tier),

            "kills": m.killer.kills,
            "deaths": m.killer.deaths,
            "assists": m.killer.assists,
            "score": m.killer.score,

            "hit_accuracy": ke.hit_accuracy,
            "time_on_target": ke.time_on_target_before_kill,
            "aim_deviation": ke.aim_deviation_before_kill,
            "reaction_time": ke.reaction_time,
            "shots_fired": ke.shots_fired,
            "shots_hit": ke.shots_hit,
            "wallbang": int(ke.wallbang),
            "kill_distance": kill_distance,

            "is_cheating": int(m.killer.is_cheating),
        })
    return pd.DataFrame(rows)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["kda"]            = (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)
    df["kd_ratio"]       = df["kills"] / df["deaths"].clip(lower=1)
    df["score_per_kill"] = df["score"] / df["kills"].clip(lower=1)


    df["sub_100ms_reaction"] = (df["reaction_time"] < 100).astype(int)
    df["sub_50ms_reaction"]  = (df["reaction_time"] < 50).astype(int)



    df["precision_score"] = df["hit_accuracy"] / df["aim_deviation"].clip(lower=0.01)


    rank_avg_kills  = df["rank_name"].map(lambda r: RANK_STAT_AVERAGES[r]["kills"][0])
    rank_std_kills  = df["rank_name"].map(lambda r: RANK_STAT_AVERAGES[r]["kills"][1])
    rank_avg_deaths = df["rank_name"].map(lambda r: RANK_STAT_AVERAGES[r]["deaths"][0])
    rank_std_deaths = df["rank_name"].map(lambda r: RANK_STAT_AVERAGES[r]["deaths"][1])
    rank_avg_score  = df["rank_name"].map(lambda r: RANK_STAT_AVERAGES[r]["score"][0])
    rank_std_score  = df["rank_name"].map(lambda r: RANK_STAT_AVERAGES[r]["score"][1])

    df["tier_relative_kills"]  = df["kills"]  - rank_avg_kills
    df["tier_relative_deaths"] = df["deaths"] - rank_avg_deaths
    df["tier_relative_score"]  = df["score"]  - rank_avg_score

    df["kills_z"]  = df["tier_relative_kills"]  / rank_std_kills
    df["deaths_z"] = df["tier_relative_deaths"] / rank_std_deaths
    df["score_z"]  = df["tier_relative_score"]  / rank_std_score

    return df



FEATURE_COLUMNS = [

    "tier",

    "kills", "deaths", "assists", "score",

    "kda", "kd_ratio", "score_per_kill",

    "hit_accuracy", "time_on_target", "aim_deviation", "reaction_time",
    "shots_fired", "shots_hit", "wallbang", "kill_distance",

    "sub_100ms_reaction", "sub_50ms_reaction", "precision_score",

    "tier_relative_kills", "tier_relative_deaths", "tier_relative_score",
    "kills_z", "deaths_z", "score_z",
]
