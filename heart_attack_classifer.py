

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# Load the data
train = pd.read_csv('/content/sample_data/train1.csv')
X_test = pd.read_csv('/content/sample_data/test1.csv')

# Split the data into X (features) and y (target)
X_train = train.drop('HeartDisease', axis=1)
y_train = train['HeartDisease']

pca = RandomizedPCA(n_components=10)
pca.fit(X_train)
svc = SVC(kernel='linear', class_weight='balanced')
model = make_pipeline(pca, svc)


param_grid = {'svc__C': [1,11],
              'svc__gamma': [0.001, 0.005 ]}

grid = GridSearchCV(model, param_grid, verbose=True) 
grid.fit(X_train, y_train)
print(grid.best_params_)

pca= RandomizedPCA(n_components=10)
svc = SVC(kernel='linear', class_weight='balanced', C=1, gamma=0.001)
model = make_pipeline(pca, svc)
model.fit(X_train, y_train)

# Make predictions on the testing dataset
y_predpcasvc = model.predict(X_test)
print(y_predpcasvc)

dfpcasvc = pd.DataFrame(y_predpcasvc, pid)
dfpcasvc.to_csv("dfpcasvc.csv")