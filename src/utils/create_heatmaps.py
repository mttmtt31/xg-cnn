import json
from tqdm import tqdm
import math
import numpy as np

# load the dictionary shot
with open('data/shots.json', 'rb') as f:
    dict_shots = json.load(f) 


# create the array of shots
all_shot_frames = np.zeros((len(dict_shots), 2, 30, 40))
# keep track of the labels (goal or non goal)
labels = []
# iterate over every shot
for i, (shot_name, shot) in enumerate(tqdm(dict_shots.items())):
    # create the heatmap of the shot
    shot_frame = np.zeros((2, 60, 40))
    # add the ball location, after having discretised its location
    ball_discrete_x = min(math.floor(shot['ball'][0] / 2), 59)
    ball_discrete_y = min(math.floor(shot['ball'][1] / 2), 39)
    shot_frame[1, ball_discrete_x, ball_discrete_y] = 1
    # iterate over every player
    for player_id, player in shot.items():
        # make sure this is actually a player
        if player_id not in ['ball', 'outcome', 'match_id', 'index']:
            # only consider opponents
            if not player['teammate']:
                # add the player in the corresponding location, after having discretised it
                discrete_x = min(math.floor(player['location'][0] / 2), 59)
                discrete_y = min(math.floor(player['location'][1] / 2), 39)
                # add the player in the right location, only if they are actually closer to the goal than the ball or if they are in a 1x1 neighbourhood around the ball
                if (discrete_x >= ball_discrete_x) or (discrete_x >= ball_discrete_x - 1 and abs(discrete_y - ball_discrete_y) <= 1):
                    shot_frame[0, discrete_x, discrete_y] += 1
    
    # all players have been added, crop out the defensive half
    cropped_shot_frame = shot_frame[:, 30:, :]
    # # add it to the global array of shots, after having smoothed it out using gaussian smoothing filter
    # all_shot_frames[i] = gaussian_filter(cropped_shot_frame, sigma = (0, .75, .75))
    # add it to the global array of shots, after having smoothed it out using gaussian smoothing filter
    all_shot_frames[i] = cropped_shot_frame
    # add the label (goal or nongoal)
    labels.append(0 if shot_name.startswith('n') else 1)

np.save('data/shots_neighbourhood.npy', all_shot_frames)
np.save('data/labels_neighbourhood.npy', labels)