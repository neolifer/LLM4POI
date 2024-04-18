import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm




def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # List to store the QA pairs
    qa_pairs = []

    # Iterate over each user
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]

        # Iterate over each unique trajectory for the user based on 'pseudo_session_trajectory_id'
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]

            # Get the start time of the current trajectory
            start_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].min()

            num_traj = user_trajectory_data.shape[-1]
            if 'traj_id' in kqt.keys():
                top200 = kqt['traj_id']
                # Fetch historical data before the start of the current trajectory
                if historical_data is not None:
                    user_historical_data = historical_data[(str(historical_data['pseudo_session_trajectory_id']) in top200)].tail(200 - num_traj)
                else:
                    user_historical_data = main_data[(str(main_data['pseudo_session_trajectory_id']) in top200)].tail(200 - num_traj)
            else:
                if historical_data is not None:
                    user_historical_data = historical_data[(historical_data['UserId'] == user) & (
                            historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)].tail(600 - num_traj)
                else:
                    user_historical_data = user_data[
                        (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)].tail(
                        600 - num_traj)
            user_trajectory_data.reset_index(drop=True, inplace=True)
            # Create the question based on the current trajectory (excluding the last entry) and historical data
            question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
            for i, row in user_trajectory_data.iloc[:-1].iterrows():
                if i > 0:
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")
                else:
                    question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")
            if not user_historical_data.empty:
                if len(user_trajectory_data.iloc[:-1]) > 0:
                    question_parts.append("There is also historical data:")
                else:
                    question_parts = [f"There is historical data for user {user}:"]
                for _, row in user_historical_data.iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")

            # Create the final question string
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # Form the answer based on the last entry of the current trajectory
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # Append the question-answer pair to the list
            qa_pairs.append((question, answer))
    return qa_pairs

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


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
    kqt1 = jload(f'{path}train_key_top200.json')
    kqt2 = jload(f'{path}test_key_top200.json')
    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args)
    qa_pairs_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)


    # Save the test QA pairs in TXT format
    with open(f'{path}test_qa_pairs_kqt.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()

