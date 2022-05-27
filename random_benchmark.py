import numpy as np
import pandas as pd

def main():
    print("Reading test data...")
    test = pd.read_csv('data/test_set_VU_DM.csv', parse_dates=['date_time'])

    print("Creating random submission...")
    np.random.seed(1)
    ordinals = np.arange(len(test))
    np.random.shuffle(ordinals)

    sorted_indices = np.argsort(ordinals)[::-1]
    prop_ids = test['prop_id'].iloc[sorted_indices]

    with open('data/random_benchmark.csv', 'w') as fout:
        fout.write("srch_id,prop_id\n")
        for srch_id, group in test.groupby(['srch_id']):
            for prop_id in prop_ids:
                fout.write(f"{srch_id},{prop_id}\n")
    print("Submission made!")

if __name__=="__main__":
    main()
