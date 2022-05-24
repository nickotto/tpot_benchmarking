from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40) 
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))