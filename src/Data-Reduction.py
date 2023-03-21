#
#   Implement Principal Component Analysis (PCA) on the iris dataset
#   to reduce the dimensionality of the data.


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_data():
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    variables = ['sepal length', 'sepal width', 'petal length', 'petal width']
    
    
    df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
    
    X = df.loc[:, variables].values
    y = df.loc[:,['target']].values
    
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    #print(df.head())
    #print(X.head())    
    
    return X, y

def plot_pca_variance(explained_variance):
    
    plt.figure(figsize=(8,6))
    plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center', label='individual explained variance')
    plt.ylabel("Variance ration")
    plt.xlabel("Principal components")
    plt.show()
    
def plot_pca_2d(df_x_pca):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    
    for target, color in zip(targets, colors):
        indicesToKeep = df_x_pca['target'] == target
        ax.scatter(df_x_pca.loc[indicesToKeep, 'PC1'], df_x_pca.loc[indicesToKeep, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    
    plt.show()

def pca(X, y):
    
    pca = PCA()
    x_pca = pca.fit_transform(X)
    df_x_pca = pd.DataFrame(x_pca)
    df_x_pca['target'] = y
    df_x_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'target']
    
    #print(df_x_pca.head())
    
    #print(df_x_pca)
    
    explained_variance = pca.explained_variance_ratio_
    
    #print(explained_variance)
    # [0.72770452 0.23030523 0.03683832 0.00515193]
    # The first and second principal components contain 95.8% of the information.
    # These two components can be used to reduce the dimensionality of the data,
    # from 4 to 2 dimensions while retaining 95.8% of the information.
    
    #plot_pca_variance(explained_variance)
    plot_pca_2d(df_x_pca)
    
def main():
    
    X, y = get_data()
    pca(X, y)
    
if __name__ == "__main__":
    main()