import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    # Load dataset
    dataset_path = os.path.join('data', 'dataset.csv')
    dataset = np.loadtxt(dataset_path, delimiter=',')

    # Split dataset into features (X) and target (y)
    X = dataset[:, :-1]
    y = dataset[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Load the trained model from the file
    with open('trained_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Make predictions on the test set
    y_pred = loaded_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))