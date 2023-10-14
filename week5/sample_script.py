import pickle

with open('./dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('./model1.bin', 'rb') as f:
    model = pickle.load(f)

res = model.predict_proba(dv.transform([{"job": "retired", "duration": 445, "poutcome": "success"}]))[0, 1]
print(res)
