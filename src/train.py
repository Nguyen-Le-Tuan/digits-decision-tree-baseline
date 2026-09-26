from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("First five label: ", y[:5])

    print("Start Experiment")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Config model: max_depth = 5, random_state = 42 with the accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred))
    

if __name__ == "__main__":
    main()