import sys

import pandas as pd
import json
import argparse

import pandas as pd
import json
from tqdm import tqdm

def generate_kq_pairs(main_data):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # List to store the QA pairs
    key_query_pairs = []

    # Iterate over each user
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]

        # Iterate over each unique trajectory for the user based on 'pseudo_session_trajectory_id'
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]
            start_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].min()
            end_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].max()

            # Create the query using the last entry in the trajectory
            query = [f"The following data is a trajectory of user {user}:"]
            for _, row in user_trajectory_data.iterrows():
                query.append(
                    f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")

            query = " ".join(query)

            # Check if current trajectory has only one entry
            if len(user_trajectory_data) == 1:
                # Get previous trajectories of the same user up to 5 entries
                prev_trajectories = user_data[user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj]
                if not prev_trajectories.empty:
                    prev_entries = prev_trajectories.tail(5)  # Get up to 5 latest entries from previous trajectories
                    key = [f"The following data is a trajectory of user {user}:"]
                    for _, row in prev_entries.iterrows():
                        key.append(
                            f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")

                    key = " ".join(key)
                # Only continue if there are entries in the trajectory
                else:
                    continue
            else:
                # Create the key based on the current trajectory
                key = [f"The following data is a trajectory of user {user}:"]
                for _, row in user_trajectory_data[:-1].iterrows():
                    key.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")

                key = " ".join(key)



            # Append the question-answer pair to the list
            key_query_pairs.append((key, query, str(traj_id), str(start_time_of_current_traj), str(end_time_of_current_traj)))

    return key_query_pairs

import re
def simplify_poi_category(text):
    # Modify this regular expression pattern based on the specific substitution you need
    return re.sub(r"\[\{'url': '[^']+', 'name': '([^']+)'}\]", r'\1', text)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process dataset names.")

    # Add an argument for the dataset name
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")

    # Parse the arguments
    args = parser.parse_args()

    # Your processing code here
    print(f"Processing dataset: {args.dataset_name}")
    path = f'../datasets/{args.dataset_name}/preprocessed/'
    # Read the data
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample_with_traj.csv')
    train_data['PoiCategoryName'] = train_data['PoiCategoryName'].apply(simplify_poi_category)

    # Save the modified DataFrame to a new CSV file
    train_data.to_csv(f'{path}train_sample.csv', index=False)
    test_data['PoiCategoryName'] = test_data['PoiCategoryName'].apply(simplify_poi_category)

    # Save the modified DataFrame to a new CSV file
    test_data.to_csv(f'{path}test_sample.csv', index=False)
    # Generate the QA pairs
    kq_pairs_train = generate_kq_pairs(train_data)
    kq_pairs_test = generate_kq_pairs(test_data)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"key": q, "query": a, "traj_id": t, 'start_time':s, 'end_time':e} for q, a, t, s, e in kq_pairs_train]
    print(len(qa_dict_train))
    with open(f'{path}train_kq_pairs.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    qa_dict_test = [{"key": q, "query": a, "traj_id": t, 'start_time':s, 'end_time':e} for q, a, t, s, e in kq_pairs_test]
    print(len(qa_dict_test))
    with open(f'{path}test_kq_pairs.json', 'w') as json_file:
        json.dump(qa_dict_test, json_file)


if __name__ == "__main__":
    main()

