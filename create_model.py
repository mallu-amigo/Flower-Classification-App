# create_model.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
from os import path, makedirs

# ensure model folder exists
model_dir = path.join(path.dirname(__file__), "model")
makedirs(model_dir, exist_ok=True)

data = load_iris()
X, y = data.data, data.target
clf = LogisticRegression(solver="liblinear", max_iter=200)
clf.fit(X, y)

# Save model + target names together
out = {"model": clf, "target_names": data.target_names.tolist()}
pkl_path = path.join(model_dir, "lr_model.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(out, f)

print("Saved model to", pkl_path)
