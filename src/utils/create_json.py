from statsbombpy import sb
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings("ignore")
import pandas as pd

# create a json
shots = {}
# initialise variables
num_shots = 0
num_goals = 0

# read competitions
df_competitions = sb.competitions()
# get rid of barcelona and invincible season
df_competitions = df_competitions[
    df_competitions.apply(
            lambda row: 
                (int(row['season_name'].split('/')[0]) >= 2010)
                & 
                ((row['competition_name'] != 'La Liga') | ((row['competition_name'] == 'La Liga') & (row['season_name'] == '2015/2016')))
                , axis = 1)
    ]

# find games
all_matches = []
for i, row in tqdm(df_competitions.iterrows(), total = len(df_competitions), desc = 'Scanning competitions to find games'):
    try:
        all_matches.extend(sb.matches(competition_id=row['competition_id'], season_id=row['season_id'])['match_id'].tolist())
    except:
        continue

print('Games have been individuated.')

df_matching = []

# keep track of how many shots
num_shots = 0
num_goals = 0
# for every game
for match_id in tqdm(all_matches, total = len(all_matches), desc = 'Scanning games, looking for shots.'):
    # extract all shots
    df_shots = sb.events(match_id, split=True)['shots']
    # isolate open-play shots
    df_shots = df_shots.loc[(~df_shots['shot_freeze_frame'].isna()) & (df_shots['shot_type'] == 'Open Play'), ['index', 'location', 'shot_freeze_frame', 'shot_outcome']].reset_index(drop = True)

    for _, shot in df_shots.iterrows():
        # create shot dictionary
        shot_dict = {str(index): dictionary for index, dictionary in enumerate(shot['shot_freeze_frame'])}
        shot_dict['ball'] = shot['location']
        shot_dict['outcome'] = shot["shot_outcome"].lower()
        if shot_dict['outcome'] == 'goal':
            shot_name = f'goals/{num_goals}'
            num_goals = num_goals + 1
        else:
            shot_name = f'non_goals/{num_shots}'
            num_shots = num_shots + 1
        shot_dict['match_id'] = match_id
        shot_dict['index'] = shot['index']
        shots[shot_name] = shot_dict
        df_matching.append([shot_name, match_id, shot['index']])

with open('data/shots.json', 'w') as f:
    json.dump(shots, f)

pd.DataFrame(df_matching, columns = ['shot_name', 'match_id', 'index']).to_csv('data/matching.csv', index = False)