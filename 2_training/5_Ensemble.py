#!/usr/bin/env python3
# Stacking ensemble method: vecstack module
from vecstack import stacking
s, st = stacking(models, X, y, Xt, regression = True, n_folds = 4, shuffle = True, random_state = 74)
allplus = svmlin.fit(s, y)
yp = allplus.predict(st)

# Accurary and recall
accuracy_score(yp, yt)
recall_score(yp, yt)

