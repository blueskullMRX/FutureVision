from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor



def get_prediction_type(model_id):
    classification = [0,1,2,3,5,6,7]
    regression = [1,3,5,6,7]
    clustering = [4]
    if model_id in classification and model_id in regression :
        return ["Classification","Regression"]
    elif model_id in classification:
        return ["Classification"]
    elif model_id in regression:
        return ["Regression"]
    elif model_id in clustering:
        return ["Clustering"]


def get_model(model_id,prediction_type):
    model = None
    if model_id == 0:
        model = LogisticRegression()
    if model_id == 1:        
        if prediction_type == "Classification":model = DecisionTreeClassifier()
        else : model = DecisionTreeRegressor()
    if model_id == 2:        
        model = GaussianNB()
    if model_id == 3:        
        if prediction_type == "Classification":model = SVC()
        else : model = SVR()
    if model_id == 4:        
        model = KMeans(n_clusters=3)
    if model_id == 5:        
        if prediction_type == "Classification":model = KNeighborsClassifier()
        else : model = KNeighborsRegressor()
    if model_id == 6:        
        if prediction_type == "Classification":model = RandomForestClassifier()
        else : model = RandomForestRegressor()
    if model_id == 7:
        if prediction_type == "Classification":model = MLPClassifier()
        else : model = MLPRegressor()
    return model