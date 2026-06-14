from dataclasses import dataclass
from typing import Any, Dict, List
import random
import math

RANK_STAT_AVERAGES = {
    "iron":      {"kills": (8,  4), "deaths": (15, 3), "assists": (4,  2), "score": (900,  200)},
    "bronze":    {"kills": (10, 4), "deaths": (13, 3), "assists": (5,  2), "score": (1100, 250)},
    "silver":    {"kills": (12, 4), "deaths": (12, 3), "assists": (6,  2), "score": (1400, 300)},
    "gold":      {"kills": (14, 4), "deaths": (10, 3), "assists": (6,  2), "score": (1800, 300)},
    "platinum":  {"kills": (16, 4), "deaths": (9,  3), "assists": (7,  2), "score": (2100, 350)},
    "diamond":   {"kills": (18, 4), "deaths": (8,  3), "assists": (8,  2), "score": (2500, 350)},
    "ascendant": {"kills": (20, 4), "deaths": (7,  3), "assists": (8,  2), "score": (2800, 400)},
    "immortal":  {"kills": (22, 4), "deaths": (6,  3), "assists": (9,  2), "score": (3200, 400)},
    "radiant":   {"kills": (25, 4), "deaths": (4,  2), "assists": (10, 2), "score": (3600, 450)},
}

RANK_NAMES = ["iron", "bronze", "silver", "gold", "platinum",
              "diamond", "ascendant", "immortal", "radiant"]

TEAMS = ["Red", "Blue"]

@dataclass
class PlayerSession:
    puuid: str               
    team_id: str             
    competitive_tier: int    
    kills: int               
    deaths: int              
    assists: int             
    score: int               
    is_cheating: bool

def tier_to_rank_name(tier: int) -> str:
    return RANK_NAMES[(tier - 1) // 3]

def gauss_int(mean: float, std: float, min_val: int = 0) -> int:
    return max(min_val, round(random.gauss(mean, std)))

def gauss_clipped(mean: float, std: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, random.gauss(mean, std)))

def generate_player_session(is_cheating: bool) -> PlayerSession:
    tier = random.randint(1, 27)
    rank_name = tier_to_rank_name(tier)
    averages = RANK_STAT_AVERAGES[rank_name]

    if is_cheating:
        inflation       = random.uniform(1.5, 2.3)
        death_deflation = random.uniform(0.3, 0.6)
        kill_mean,   kill_std   = averages["kills"]
        death_mean,  death_std  = averages["deaths"]
        assist_mean, assist_std = averages["assists"]
        score_mean,  score_std  = averages["score"]
        kills   = gauss_int(kill_mean   * inflation,       kill_std   * inflation, min_val=1)
        deaths  = gauss_int(death_mean  * death_deflation, death_std,              min_val=0)
        assists = gauss_int(assist_mean * inflation,       assist_std * inflation, min_val=0)
        score   = gauss_int(score_mean  * inflation,       score_std  * inflation, min_val=0)
    else:
        kills   = gauss_int(*averages["kills"],   min_val=0)
        deaths  = gauss_int(*averages["deaths"],  min_val=1)
        assists = gauss_int(*averages["assists"],  min_val=0)
        score   = gauss_int(*averages["score"],   min_val=0)

    return PlayerSession(
        puuid            = f"player_{random.randint(10000, 99999)}",
        team_id          = random.choice(TEAMS),
        competitive_tier = tier,
        kills            = kills,
        deaths           = deaths,
        assists          = assists,
        score            = score,
        is_cheating      = is_cheating,
    )
    

@dataclass
class KillEvent:
    kill_time_in_round: int
    killer_fov_enter_time: float
    killer_team: str
    victim_team: str
    victim_death_location: Dict[str, float]
    damage_weapon_name: str
    player_locations_on_kill: List[Dict[str, Any]]
    time_on_target_before_kill: float
    hit_accuracy: float
    aim_deviation_before_kill: float
    shots_fired: int
    shots_hit: int
    reaction_time: float
    wallbang: bool = False
    def __post_init__(self):
        self.hit_accuracy = self.shots_hit / self.shots_fired


@dataclass
class MatchSession:
    duration: int
    killer: PlayerSession
    victim: PlayerSession
    kill_event: KillEvent


MAP_COORDINATE_RANGE = (-4000, 4000)
MAX_LOCATION_DISTANCE = 1000


def random_location() -> dict:
    return {
        "x": round(random.uniform(*MAP_COORDINATE_RANGE), 1),
        "y": round(random.uniform(*MAP_COORDINATE_RANGE), 1),
    }


def clamp_location(location: dict) -> dict:
    return {
        "x": min(max(location["x"], MAP_COORDINATE_RANGE[0]), MAP_COORDINATE_RANGE[1]),
        "y": min(max(location["y"], MAP_COORDINATE_RANGE[0]), MAP_COORDINATE_RANGE[1]),
    }


def nearby_location(reference_location: dict, max_distance: int = MAX_LOCATION_DISTANCE) -> dict:
    angle = random.uniform(0, 2 * math.pi)
    distance = random.uniform(0, max_distance)
    location = {
        "x": round(reference_location["x"] + math.cos(angle) * distance, 1),
        "y": round(reference_location["y"] + math.sin(angle) * distance, 1),
    }
    return clamp_location(location)


def killer_victim_locations(max_distance: int = MAX_LOCATION_DISTANCE) -> tuple[dict, dict]:
    killer_location = random_location()
    victim_location = nearby_location(killer_location, max_distance=max_distance)
    return killer_location, victim_location


def make_player_location(team: str, location: dict) -> dict:
    return {
        "team": team,
        "location": location,
    }


def build_kill_event(
    killer_team: str,
    victim_team: str,
    is_cheating: bool,
    cheater_hold_time_mean: float = 40,
    cheater_hold_time_std: float = 20,
) -> KillEvent:



    if is_cheating:
        hold_time     = gauss_clipped(cheater_hold_time_mean, cheater_hold_time_std, 10, 500)
        aim_deviation = gauss_clipped(1.8, 0.7, 0.4, 4.0)
        shots_fired   = random.randint(1, 3)
        shots_hit     = random.randint(max(1, shots_fired - 1), shots_fired)
    else:
        hold_time     = gauss_clipped(280, 80,  80, 500)
        aim_deviation = gauss_clipped(1.8, 0.7, 0.4, 4.0)
        shots_fired   = random.randint(1, 10)
        shots_hit     = random.randint(1, shots_fired)
    fov_enter_time = random.uniform(50, 300)
    killer_location, victim_location = killer_victim_locations()
    reaction_time = round(fov_enter_time + hold_time, 1)

    return KillEvent(
        kill_time_in_round=int(random.uniform(1000, 100000)),
        killer_fov_enter_time=round(fov_enter_time, 1),
        killer_team=killer_team,
        victim_team=victim_team,
        victim_death_location=victim_location,
        damage_weapon_name="Vandal",
        player_locations_on_kill=[
            make_player_location(killer_team, killer_location),
            make_player_location(victim_team, victim_location),
        ],
        time_on_target_before_kill=round(hold_time, 1),
        hit_accuracy=0.0,
        aim_deviation_before_kill=round(aim_deviation, 3),
        shots_fired=shots_fired,
        shots_hit=shots_hit,
        reaction_time=reaction_time
    )
    