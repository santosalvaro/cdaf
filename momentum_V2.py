#instalar a biblioteca do statsbombpy
!pip install statsbombpy



#lista as competiçoes disponiveis
from statsbombpy import sb

df = sb.competitions()
df[['competition_name', 'season_name']].drop_duplicates().sort_values(['competition_name', 'season_name'])

#escolhendo competiçao e time
sb.competitions().query('competition_name == "1. Bundesliga" & season_name == "2023/2024"')

df = sb.matches(competition_id=9, season_id=281)

df = df.query('home_team == "Bayer Leverkusen"')

print(df[['home_team', 'away_team', 'home_score', 'away_score']])

# selecionando (de fato) a partidade
MATCH_ID = 3895052
HOME_TEAM, AWAY_TEAM = 'Bayer Leverkusen', 'RB Leipzig'
df = sb.events(match_id=MATCH_ID)



from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def momentum(match_id, window_size=4, decay_rate=0.25, sigma=1):
    df = sb.events(match_id=match_id)
    HOME_TEAM, AWAY_TEAM= list(df['team'].unique())

    xT = pd.read_csv("https://raw.githubusercontent.com/AKapich/WorldCup_App/main/app/xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape

   #Verificar a corretude desta função*
    def get_xT(df, event_type):
        df = df[df['type']==event_type]

        df['x'], df['y'] = zip(*df['location'])
        df['end_x'], df['end_y'] = zip(*df[f'{event_type.lower()}_end_location'])

        df[f'start_x_bin'] = pd.cut(df['x'], bins=xT_cols, labels=False)
        df[f'start_y_bin'] = pd.cut(df['y'], bins=xT_rows, labels=False)
        df[f'end_x_bin'] = pd.cut(df['end_x'], bins=xT_cols, labels=False)
        #Verificar essa parte de end_y_bin usar end_x*
        #talvez deveria ser:
        #df[f'end_y_bin'] = pd.cut(df['end_y'], bins=xT_rows, labels=False)
        #ao inves de :
        df[f'end_y_bin'] = pd.cut(df['end_x'], bins=xT_rows, labels=False)
        df['start_zone_value'] = df[[f'start_x_bin', f'start_y_bin']].apply(lambda z: xT[z[1]][z[0]], axis=1)
        df['end_zone_value'] = df[[f'end_x_bin', f'end_y_bin']].apply(lambda z: xT[z[1]][z[0]], axis=1)
        df['xT'] = df['end_zone_value']-df['start_zone_value']

        return df[['xT', 'minute', 'second', 'team', 'type']]


    xT_data = pd.concat([get_xT(df=df, event_type='Pass'), get_xT(df=df, event_type='Carry')], axis=0)
    xT_data['xT_clipped'] = np.clip(xT_data['xT'], 0, 0.1)

    max_xT_per_minute = xT_data.groupby(['team', 'minute'])['xT_clipped'].max().reset_index()

    minutes = sorted(xT_data['minute'].unique())
    weighted_xT_sum = {team: [] for team in max_xT_per_minute['team'].unique()}
    momentum = []

    for current_minute in minutes:
        for team in weighted_xT_sum:
            recent_xT_values = max_xT_per_minute[(max_xT_per_minute['team'] == team) &
                                                    (max_xT_per_minute['minute'] <= current_minute) &
                                                    (max_xT_per_minute['minute'] > current_minute - window_size)]

            weights = np.exp(-decay_rate * (current_minute - recent_xT_values['minute'].values))
            weighted_sum = np.sum(weights * recent_xT_values['xT_clipped'].values)
            weighted_xT_sum[team].append(weighted_sum)

        momentum.append(weighted_xT_sum[HOME_TEAM][-1] - weighted_xT_sum[AWAY_TEAM][-1])

    momentum_df = pd.DataFrame({
        'minute': minutes,
        'momentum': momentum
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([0,15,30,45,60,75,90])
    ax.margins(x=0)
    ax.set_ylim(-0.08, 0.08)

    momentum_df['smoothed_momentum'] = gaussian_filter1d(momentum_df['momentum'], sigma=sigma)
    ax.plot(momentum_df['minute'], momentum_df['smoothed_momentum'], color='white')

    ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
    ax.fill_between(momentum_df['minute'], momentum_df['smoothed_momentum'], where=(momentum_df['smoothed_momentum'] > 0), color='blue', alpha=0.5, interpolate=True)
    ax.fill_between(momentum_df['minute'], momentum_df['smoothed_momentum'], where=(momentum_df['smoothed_momentum'] < 0), color='red', alpha=0.5, interpolate=True)

    scores = df[df['shot_outcome'] == 'Goal'].groupby('team')['shot_outcome'].count().reindex(set(df['team']), fill_value=0)
    ax.set_xlabel('Minute', color='white', fontsize=15, fontweight='bold', fontfamily='Monospace')
    ax.set_ylabel('Momentum', color='white', fontsize=15, fontweight='bold', fontfamily='Monospace')
    ax.set_title(f'xT Momentum\n{HOME_TEAM} {scores[HOME_TEAM]}-{scores[AWAY_TEAM]} {AWAY_TEAM}', color='white', fontsize=20, fontweight='bold', fontfamily='Monospace', pad=-5)

    home_team_text = ax.text(7, 0.064, HOME_TEAM, fontsize=12, ha='center', fontfamily="Monospace", fontweight='bold', color='white')
    home_team_text.set_bbox(dict(facecolor='blue', alpha=0.5, edgecolor='white', boxstyle='round'))
    away_team_text = ax.text(7, -0.064, AWAY_TEAM, fontsize=12, ha='center', fontfamily="Monospace", fontweight='bold', color='white')
    away_team_text.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='white', boxstyle='round'))

    goals = df[df['shot_outcome']=='Goal'][['minute', 'team']]
    for _, row in goals.iterrows():
        ymin, ymax = (0.5, 0.8) if row['team'] == HOME_TEAM else (0.14, 0.5)
        ax.axvline(row['minute'], color='white', linestyle='--', linewidth=0.8, alpha=0.5, ymin=ymin, ymax=ymax)
        ax.scatter(row['minute'], (1 if row['team'] == HOME_TEAM else -1)*0.06, color='white', s=100, zorder=10, alpha=0.7)
        ax.text(row['minute']+0.1, (1 if row['team'] == HOME_TEAM else -1)*0.067, 'Goal', fontsize=10, ha='center', va='center', fontfamily="Monospace", color='white')



momentum(match_id=MATCH_ID)
