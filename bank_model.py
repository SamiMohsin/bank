import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

test_size = 0.2
random_state = 42

if __name__ == '__main__':
    feats = pd.read_csv('data/feats_e3.csv', index_col=0)
    target = pd.read_csv('data/target_e3.csv', index_col=0)

    x_train, x_test, y_train, y_test = train_test_split(feats, target, test_size=test_size, random_state=random_state)

    print(f'X_tarin: {x_train.shape}')
    print(f'X_test: {x_test.shape}')
    print(f'Y_tarin: {y_train.shape}')
    print(f'Y_test: {y_test.shape}')

    model = LogisticRegression(random_state=random_state, solver='liblinear')
    model.fit(x_train, y_train['y'])

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)

    print(f'Accuracy = {accuracy * 100:0.4f}%')

    # save result to new csv
    bank_y = pd.DataFrame(data=y_pred, columns=['y'])
    print(bank_y)


    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average='binary')
    print(f'precision: {precision}\nrecall: {recall}\nfscore: {fscore}\n')

    coef_list = [f'{coef}: {val}' for coef, val in sorted(zip(model.coef_[0], x_train.columns.values))]

    for item in coef_list:
        print(item)



