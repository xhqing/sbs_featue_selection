sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in sss.split(data, data["month"]):
    train = data.iloc[train_index]
    test = data.iloc[test_index]

train_x, train_y = train.drop("dpc", axis=1), train["dpc"]
test_x, test_y = test.drop("dpc", axis=1), test["dpc"]

remaining_feas = train_x.columns.tolist()
dropped_feas = []

def target_func(remaining_feas: list, current_fea: str):
    lr = LinearRegression()
    ss = StandardScaler()
    if current_fea:
        remaining_feas.remove(current_fea)
    ss.fit(train_x[remaining_feas])
    lr.fit(ss.transform(train_x[remaining_feas]), train_y)
    
    score = mean_absolute_error(test_y, lr.predict(test_x[remaining_feas]))
    return score

def fea_selection(remaining_feas: list, dropped_feas: list, score_target: float=1000.0):
    candidate_drop_fea = None
    mae_score = target_func(remaining_feas, None)
    Break = False
    remaining_feas_copy = remaining_feas.copy()
    for fea in remaining_feas:
        current_score = target_func(remaining_feas_copy, fea)
        if current_score < mae_score:
            print(f"update candidate drop fea: {fea}, current mae score: {current_score}")
            candidate_drop_fea = fea
            mae_score = current_score
            if mae_score <= score_target:
                Break = True
                break
    
    
    dropped_feas.append(candidate_drop_fea)
    remaining_feas.remove(candidate_drop_fea)
    return dropped_feas, remaining_feas, Break


for i in range(len(remaining_feas)):
    print("-------- loop {}: ".format(i))
    dropped_feas, remaining_feas, Break = fea_selection(remaining_feas, dropped_feas, score_target=1670)
    if Break: break
