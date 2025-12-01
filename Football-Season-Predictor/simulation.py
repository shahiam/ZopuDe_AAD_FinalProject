import numpy as np
import pandas as pd


#core strength model
def estimate_strengths(historical_matches):
    hist = historical_matches.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
    total_matches = len(hist)
    if total_matches == 0:
        raise ValueError("No completed matches to estimate strengths.")

    league_avg_goals = (hist["HomeGoals"].sum() + hist["AwayGoals"].sum()) / (2 * total_matches)
    teams = pd.Index(sorted(set(hist["HomeTeam"]) | set(hist["AwayTeam"])))
    team_stats = pd.DataFrame(index=teams, columns=["GF", "GA", "MP"], dtype=float).fillna(0.0)

    for _, r in hist.iterrows():
        ht, at = r["HomeTeam"], r["AwayTeam"]
        hg, ag = float(r["HomeGoals"]), float(r["AwayGoals"])
        team_stats.loc[ht, "GF"] += hg
        team_stats.loc[ht, "GA"] += ag
        team_stats.loc[ht, "MP"] += 1
        team_stats.loc[at, "GF"] += ag
        team_stats.loc[at, "GA"] += hg
        team_stats.loc[at, "MP"] += 1

    team_stats["GFpm"] = team_stats["GF"] / team_stats["MP"].replace(0, np.nan)
    team_stats["GApm"] = team_stats["GA"] / team_stats["MP"].replace(0, np.nan)
    team_stats["attack"] = team_stats["GFpm"] / league_avg_goals
    team_stats["defense"] = team_stats["GApm"] / league_avg_goals
    team_stats["attack"].fillna(1.0, inplace=True)
    team_stats["defense"].fillna(1.0, inplace=True)

    home_goals = hist["HomeGoals"].sum()
    away_goals = hist["AwayGoals"].sum()
    home_adv = (home_goals / (home_goals + away_goals)) / 0.5
    home_adv = float(np.clip(home_adv, 1.02, 1.2))

    return team_stats[["attack", "defense"]].to_dict(orient="index"), league_avg_goals, home_adv


#simulating a match
def match_lambdas(home, away, strengths, league_avg_goals, home_adv):
    a_h = strengths.get(home, {"attack": 1.0})["attack"]
    d_h = strengths.get(home, {"defense": 1.0})["defense"]
    a_a = strengths.get(away, {"attack": 1.0})["attack"]
    d_a = strengths.get(away, {"defense": 1.0})["defense"]
    lam_home = league_avg_goals * home_adv * a_h * d_a
    lam_away = league_avg_goals * a_a * d_h
    return lam_home, lam_away


def simulate_season(fixtures_df, strengths, league_avg_goals, home_adv, rng):
    teams = sorted(set(fixtures_df["HomeTeam"]) | set(fixtures_df["AwayTeam"]))
    pts = dict.fromkeys(teams, 0)
    gf = dict.fromkeys(teams, 0)
    ga = dict.fromkeys(teams, 0)

    for _, row in fixtures_df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if pd.notna(row.get("HomeGoals")) and pd.notna(row.get("AwayGoals")):
            hg, ag = int(row["HomeGoals"]), int(row["AwayGoals"])
        else:
            lam_h, lam_a = match_lambdas(home, away, strengths, league_avg_goals, home_adv)
            hg = rng.poisson(lam_h)
            ag = rng.poisson(lam_a)

        gf[home] += hg
        ga[home] += ag
        gf[away] += ag
        ga[away] += hg

        if hg > ag:
            pts[home] += 3
        elif hg < ag:
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1

    return pts, gf, ga


def montesimulation(historical_df, fixtures_df, num_simulations):
    strengths, league_avg_goals, home_adv = estimate_strengths(historical_df)
    teams = sorted(set(fixtures_df["HomeTeam"]) | set(fixtures_df["AwayTeam"]))

    total_points = {t: 0 for t in teams}
    total_gf = {t: 0 for t in teams}
    total_ga = {t: 0 for t in teams}

    rng = np.random.default_rng()

    for _ in range(num_simulations):
        pts, gf, ga = simulate_season(fixtures_df, strengths, league_avg_goals, home_adv, rng)
        for t in teams:
            total_points[t] += pts[t]
            total_gf[t] += gf[t]
            total_ga[t] += ga[t]

    table = []
    for t in teams:
        table.append({
            "Team": t,
            "Avg Points": round(total_points[t] / num_simulations, 2),
            "Avg GD": round((total_gf[t] - total_ga[t]) / num_simulations, 2),
            "Avg GF": round(total_gf[t] / num_simulations, 2),
            "Avg GA": round(total_ga[t] / num_simulations, 2),
        })

    df = pd.DataFrame(table).sort_values(by="Avg Points", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df
