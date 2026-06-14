import random
from classes import build_kill_event, MatchSession, generate_player_session

def generate_match_session(
    cheater_hold_time_mean: float = 40,
    cheater_hold_time_std: float = 20,
) -> MatchSession:
    """Generate one match (1 killer + 1 victim, mutually exclusive cheater).

    cheater_hold_time_{mean,std} pass through to build_kill_event for the
    robustness sweep (default = standard triggerbot distribution).
    """
    killer_is_cheating = random.choice([True, False])
    victim_is_cheating = not killer_is_cheating

    killer = generate_player_session(killer_is_cheating)
    victim = generate_player_session(victim_is_cheating)


    if killer.team_id == victim.team_id:
        victim.team_id = "Blue" if killer.team_id == "Red" else "Red"

    kill_event = build_kill_event(
        killer_team=killer.team_id,
        victim_team=victim.team_id,
        is_cheating=killer_is_cheating,
        cheater_hold_time_mean=cheater_hold_time_mean,
        cheater_hold_time_std=cheater_hold_time_std,
    )

    duration = random.randint(1000, 100000)

    return MatchSession(
        duration=duration,
        killer=killer,
        victim=victim,
        kill_event=kill_event,
    )

