import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # assignment description: The command line should take filenames for the labelled, unlabelled, and output files:
    # python3 weather_city.py monthly-data-labelled.csv monthly-data-unlabelled.csv labels.csv

    labelled = pd.read_csv(sys.argv[1])
    unlabelled = pd.read_csv(sys.argv[2])

    #print(labelled.head())

    # removign city cause they are the answer
    # removing year cause dont want model to learn from it
    X = labelled.drop(columns=['city', 'year']).to_numpy()
    y = labelled['city'].to_numpy(dtype=str)

    X_unlabelled = unlabelled.drop(columns=['city', 'year']).to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # a lot different scales
    scale = StandardScaler()
    model = make_pipeline(
        scale,
        SVC(kernel='linear', C=0.1)
    )
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))

    predictions = model.predict(X_unlabelled)
    pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)

    # mistakes on validation data: 
    # df = pd.DataFrame({'truth': y_valid, 'prediction': model.predict(X_valid)})
    # print(df[df['truth'] != df['prediction']])


if __name__ == '__main__':
    main()
