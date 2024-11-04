import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import root_mean_squared_error,make_scorer
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
train=pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Regression Of Used Car Prices\train.csv")
test=pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Regression Of Used Car Prices\test.csv")
print(train.info())
# print(train.dtypes)
# print(test.dtypes)
# X_train=train.drop(columns=['id','price'])
# Test=test.drop(columns=['id'])
# Y_train=train['price']

# #Identify categorical and numerical 
# nc=X_train.select_dtypes(include=np.number).columns
# cc=X_train.select_dtypes(exclude=np.number).columns
#     #print(cc)
# #Impute
# X_train[nc]=X_train[nc].fillna(X_train[nc].mean())
# Test[nc]=Test[nc].fillna(Test[nc].mean())
# X_train[cc]=X_train[cc].fillna(X_train[cc].mode().iloc[0])
# Test[cc]=Test[cc].fillna(Test[cc].mode().iloc[0])
# #encoding 
# X_train=pd.get_dummies(X_train,columns=cc,drop_first=True)
# Test=pd.get_dummies(Test,columns=cc,drop_first=True)

# Test=Test.reindex(columns=X_train.columns,fill_value=0)

# # Scaling 
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# Test=sc.fit_transform(Test)

# X_tr,X_val,Y_tr,Y_val=train_test_split(X_train,Y_train,test_size=0.1,random_state=42)

# ml=XGBRegressor(objective='reg:squarederror',n_estimators=100,learning_rate=0.1,device='cuda:0')
# ml.fit(X_tr,Y_tr)
# pred_val=ml.predict(X_val)
# print(f"error : {root_mean_squared_error(pred_val,Y_val)}")

# r_scorer=make_scorer(root_mean_squared_error)
# cv=cross_val_score(ml,X_train,Y_train,cv=5,scoring=r_scorer)
# print(f"cv mean : {cv.mean()}")
# pred=ml.predict(Test)

# # file submsiion 
# sub=pd.DataFrame({
#     'id':test['id'],'price':pred
# })
# sub.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Regression Of Used Car Prices\sample_submission.csv",index=False)
# print("ok")