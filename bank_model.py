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


"""
precision: 0.6190476190476191
recall: 0.2653061224489796
fscore: 0.37142857142857144

-1.047236345996167: job_entrepreneur
-0.8323690457593566: job_blue-collar
-0.8006973648490253: poutcome_failure
-0.6990479112660989: is_loan
-0.5881293112252126: job_services
-0.5375888889003666: job_unemployed
-0.5035124411366861: job_technician
-0.44990725040508767: marital_married
-0.3545448688662423: job_self-employed
-0.2661378493098897: job_housemaid
-0.229386303236445: job_management
-0.2276773287211516: job_admin.
-0.13343827766345473: marital_single
-0.07335046590596678: campaign
-0.06545272273621247: education_primary
-0.027235058643891648: poutcome_other
2.128675823992884e-06: balance
0.0023385676497980103: day
0.003916764386594083: duration
0.00639197280541005: month
0.006920347648190091: age
0.01179678967598105: previous
0.09130396855452388: education_secondary
0.3637152188133593: job_student
0.41773831829928176: education_tertiary
0.4561162449784006: job_retired
0.5300165340570421: is_default
0.9780127500807843: was_contact
1.8059451735736558: poutcome_success

"""
