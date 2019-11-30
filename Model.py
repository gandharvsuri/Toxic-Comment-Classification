
# Required library imports
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

from features_extraction import get_features


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def pr(y_i, y,train_features):
    p = train_features[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def EasyEnsembleClassfier(data,test_data):

    train_text = data['comment_text'].values.astype(str)
    test_text = test_data['comment_text'].values.astype(str)
    all_text = np.concatenate([train_text, test_text])
    train_features,test_features = get_features(train_text,test_text,all_text)

    submission = pd.DataFrame.from_dict({'Id': test_data['id']})
    for class_name in class_names:
        train_target = data[class_name]
        y = train_target.values
        r = np.log(pr(1,y,train_features) / pr(0,y,train_features))
        x_nb = train_features.multiply(r)
        l = EasyEnsembleClassifier(base_estimator=LogisticRegression(C=2, solver='sag', max_iter=500))
        n = EasyEnsembleClassifier(base_estimator=SGDClassifier(alpha=.0002, max_iter=180, penalty="l2", loss='modified_huber'))
        o = LogisticRegression(C=2, dual=True, max_iter=500)
        p = RandomForestClassifier(criterion='gini',
                max_depth=100, max_features=1000, max_leaf_nodes=None, min_samples_split=10,
                min_weight_fraction_leaf=0.0, n_estimators=80)  
        m = VotingClassifier(estimators=[ ('lr', l), ('sgd', n),('lr1',o),('rdf',p)], voting='soft', weights=[0.9,1.35,0.65,0.8])
        m.fit(x_nb, y)

        submission[class_name] = m.predict_proba(test_features.multiply(r))[:, 1]

        submission.to_csv('EnsembleClassfierSubmission_2.csv', index=False)
        joblib.dump(m,'Ensemble.pkl')

if __name__ == "__main__":

    data = pd.read_csv('./Data/ppc_train.csv')
    test_data = pd.read_csv('./Data/ppc_test.csv')
    EasyEnsembleClassfier(data,test_data)