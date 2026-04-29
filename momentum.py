# Instalação das bibliotecas
!pip install -q \
    gandula==1.0.0 \
    numpy \
    orjson \
    pandas \
    matplotlib \
    scikit-learn \
    statsmodels \
    scipy \
    mplsoccer \
    ipykernel \
    tqdm

import numpy as np
import pandas as pd
import orjson
from pathlib import Path
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

PITCH_LENGTH = 105
PITCH_WIDTH = 68

NX, NY = 16, 12
XT_MAX_ITERS = 100

DATA_PATH = "/content/dados"

def load_raw(path):
    files = sorted(Path(path).glob("*.json"))
    dfs = []

    for f in tqdm(files, desc="Loading"):
        data = orjson.loads(f.read_bytes())
        df = pd.DataFrame(data)

        # extrai bola
        def get_ball_xy(ball):
            if isinstance(ball, list) and len(ball) > 0:
                b = ball[0]
                return b.get("X"), b.get("Y")
            return np.nan, np.nan

        xy = df["BALL"].apply(get_ball_xy)
        df["x"] = xy.apply(lambda v: v[0])
        df["y"] = xy.apply(lambda v: v[1])

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def build_actions(df, min_move=1.0):

    df = df.sort_values("EVENT_TIME").reset_index(drop=True)

    # próxima posição
    df["x_next"] = df["x"].shift(-1)
    df["y_next"] = df["y"].shift(-1)

    # distância da bola
    df["dist"] = np.sqrt((df["x_next"] - df["x"])**2 + (df["y_next"] - df["y"])**2)

    # filtra apenas movimentos relevantes
    df = df[df["dist"] > min_move]

    # cria ações
    actions = pd.DataFrame({
        "x": df["x"],
        "y": df["y"],
        "x_end": df["x_next"],
        "y_end": df["y_next"],
        "time": df["EVENT_TIME"]
    })

    # minuto
    actions["minute"] = (actions["time"] // 60).astype(int)

    return actions.dropna()

def assign_teams(actions):

    # alterna posse quando há "grande mudança"
    actions["team_id"] = 0

    current_team = 0
    teams = []

    for i in range(len(actions)):
        if i == 0:
            teams.append(current_team)
            continue

        # heurística: mudança brusca → troca de posse
        dx = np.sqrt(
            (actions.iloc[i]["x"] - actions.iloc[i-1]["x_end"])**2 +
            (actions.iloc[i]["y"] - actions.iloc[i-1]["y_end"])**2
        )

        if dx > 10:  # hiperparâmetro
            current_team = 1 - current_team

        teams.append(current_team)

    actions["team_id"] = teams

    return actions

def classify_actions(actions):

    actions = actions.copy()

    actions["action_type"] = "carry"

    # regra 1: zona perigosa
    mask_zone = (
        (actions["x_end"] > 85) &
        (actions["y_end"].between(15, 53))
    )

    # regra 2: dinâmica da bola
    dx = actions["x_end"] - actions["x"]
    dy = actions["y_end"] - actions["y"]

    actions["speed"] = np.sqrt(dx**2 + dy**2)
    actions["speed_next"] = actions["speed"].shift(-1)

    mask_speed = (
        (actions["speed"] > 1.5) &
        (actions["speed_next"] < 0.3)
    )

    # aplica shot
    actions.loc[mask_zone | mask_speed, "action_type"] = "shot"

    return actions

def normalize(actions):

    # assume escala original ~ 0–100 (padrão tracking)
    actions["x"] = actions["x"] / 100 * 105
    actions["y"] = actions["y"] / 100 * 68

    actions["x_end"] = actions["x_end"] / 100 * 105
    actions["y_end"] = actions["y_end"] / 100 * 68

    return actions

def clamp_field(actions):

    actions["x"] = actions["x"].clip(0, 105)
    actions["y"] = actions["y"].clip(0, 68)

    actions["x_end"] = actions["x_end"].clip(0, 105)
    actions["y_end"] = actions["y_end"].clip(0, 68)

    return actions

def get_bins(x, y):

    # discretização
    bx = np.floor(x / 105 * NX).astype(int)
    by = np.floor(y / 68 * NY).astype(int)

    bx = np.clip(bx, 0, NX - 1)
    by = np.clip(by, 0, NY - 1)

    return bx, by

def fit_xt(actions):

    actions = actions.copy()

    # bins início
    actions["bx"], actions["by"] = get_bins(actions["x"].values, actions["y"].values)

    total = np.zeros((NY, NX))
    move = np.zeros((NY, NX))
    shot = np.zeros((NY, NX))
    trans = np.zeros((NY, NX, NY, NX))

    # contagens
    for _, r in actions.iterrows():

        y, x = int(r.by), int(r.bx)
        total[y, x] += 1

        if r.action_type == "shot":
            shot[y, x] += 1
        else:
            move[y, x] += 1

            by2, bx2 = get_bins(
                np.array([r.x_end]),
                np.array([r.y_end])
            )

            by2, bx2 = int(by2[0]), int(bx2[0])

            if 0 <= by2 < NY and 0 <= bx2 < NX:
                trans[y, x, by2, bx2] += 1

    # probabilidades
    P_shot = np.divide(shot, total, out=np.zeros_like(shot), where=total > 0)
    P_move = np.divide(move, total, out=np.zeros_like(move), where=total > 0)

    # probabilidade de gol
    def goal_prob(x, y):
        goal_x, goal_y = 105, 34
        dist = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        return np.exp(-dist / 20)  # decaimento suave

    P_goal = np.zeros((NY, NX))

    for iy in range(NY):
        for ix in range(NX):
            px = (ix + 0.5) / NX * 105
            py = (iy + 0.5) / NY * 68
            P_goal[iy, ix] = goal_prob(px, py)

    # normaliza transição
    for y in range(NY):
        for x in range(NX):
            s = trans[y, x].sum()
            if s > 0:
                trans[y, x] /= s

    # value iteration
    # inicializa com valor não-zero
    xT = P_goal.copy()

    for _ in range(XT_MAX_ITERS):

        move_payoff = np.einsum("ijkl,kl->ij", trans, xT)

        new_xT = P_shot * P_goal + P_move * move_payoff

        # critério de convergência
        if np.max(np.abs(new_xT - xT)) < 1e-5:
            break

        xT = new_xT

    return xT

def apply_xt(actions, xt):

    bx, by = get_bins(actions["x"], actions["y"])
    bx2, by2 = get_bins(actions["x_end"], actions["y_end"])

    actions["xT"] = xt[by2, bx2] - xt[by, bx]

    return actions
    
def compute_momentum(actions, window=4, decay=0.3):

    df = actions.copy()

    # usa valor absoluto limitado (igual Opta)
    df["xT_pos"] = df["xT"].clip(-0.1, 0.1)

    # pega máximo por minuto (não soma)
    per_min = (
        df.groupby(["team_id", "minute"])["xT_pos"]
        .max()
        .reset_index()
    )

    teams = per_min["team_id"].unique()
    t1, t2 = teams[:2]

    minutes = sorted(per_min["minute"].unique())

    momentum = []

    for m in minutes:

        def threat(team):
            sub = per_min[
                (per_min.team_id == team)
                & (per_min.minute <= m)
                & (per_min.minute > m - window)
            ]

            if len(sub) == 0:
                return 0

            weights = np.exp(-decay * (m - sub.minute))
            return np.sum(sub.xT_pos * weights)

        momentum.append(threat(t1) - threat(t2))

    return minutes, gaussian_filter1d(momentum, sigma=1.2)

def add_goals_overlay(actions):

    goals = actions[actions["action_type"] == "shot"]

    for _, g in goals.iterrows():
        plt.axvline(g["minute"], linestyle="--", alpha=0.3)

def orient_playing_direction(actions):

    # assume que time 0 ataca esquerda->direita
    # time 1 invertido

    mask = actions["team_id"] == 1

    actions.loc[mask, "x"] = 105 - actions.loc[mask, "x"]
    actions.loc[mask, "x_end"] = 105 - actions.loc[mask, "x_end"]

    actions.loc[mask, "y"] = 68 - actions.loc[mask, "y"]
    actions.loc[mask, "y_end"] = 68 - actions.loc[mask, "y_end"]

    return actions

df = load_raw(DATA_PATH)

actions = build_actions(df)

actions = actions.dropna(subset=["x", "y", "x_end", "y_end"])

actions = normalize(actions)
actions = clamp_field(actions)

actions = classify_actions(actions)
actions = assign_teams(actions)
actions = orient_playing_direction(actions)

# sanity check
print(actions["action_type"].value_counts())

xt = fit_xt(actions)
actions = apply_xt(actions, xt)

print(actions["xT"].describe())

minutes, momentum = compute_momentum(actions)

def plot_momentum_pro(minutes, momentum):

    import numpy as np
    import matplotlib.pyplot as plt

    momentum = np.array(momentum)

    plt.figure(figsize=(14, 6))

    # cores
    home_color = "#2a6fdb"   
    away_color = "#e63946"    

    # normaliza escala
    max_abs = max(abs(momentum)) if len(momentum) > 0 else 1
    momentum = momentum / max_abs * 0.1  # escala padrão visual

    # time dominante
    plt.fill_between(
        minutes, momentum, 0,
        where=(momentum >= 0),
        interpolate=True,
        color=home_color,
        alpha=0.8,
        label="Time A"
    )

    # adversário
    plt.fill_between(
        minutes, momentum, 0,
        where=(momentum < 0),
        interpolate=True,
        color=away_color,
        alpha=0.8,
        label="Time B"
    )

    # linha suave
    plt.plot(minutes, momentum, color="#1a1a1a", linewidth=1.3)

    # baseline
    plt.axhline(0, color="black", linewidth=1)

    # estética estilo broadcast
    plt.xlim(min(minutes), max(minutes))
    plt.ylim(-0.11, 0.11)

    plt.xlabel("Minuto", fontsize=11)
    plt.ylabel("Momentum", fontsize=11)
    plt.title("Match Momentum (xT Flow)", fontsize=14)

    # grid leve
    plt.grid(alpha=0.2)

    # remove bordas
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend()
    plt.tight_layout()
    plt.show()

minutes, momentum = compute_momentum(actions)

plot_momentum_pro(minutes, momentum)
add_goals_overlay(actions)
