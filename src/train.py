from sklearn.datasets import load_digits

def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(X.shape)
    print(y.shape)
    print("First five label: ", y[:5])

    print("Start Experiment")
    

if __name__ == "__main__":
    main()