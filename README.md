# -import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1.读取数据
df = pd.read_csv("telecom_churn.csv")
print("原始数据shape", df.shape)

# ========== 2 数据清洗 ==========
# 2.1 清洗分类字段
def clean_contract(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if "month" in s:
        return "Monthtomonth"
    elif "one" in s:
        return "Oneyear"
    elif "two" in s:
        return "Twoyear"
    return s

def clean_bool_val(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().upper()
    if s in ["Y","YES"]:
        return "Yes"
    elif s in ["N","NO"]:
        return "No"
    return s

df["contract_clean"] = df["contract"].apply(clean_contract)

bool_cols = ["partner","dependents","phone_service","paperless_billing"]
for col in bool_cols:
    df[col] = df[col].apply(clean_bool_val)

# 2.2清洗金额字段
def clean_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("$","").replace(",","").strip()
    try:
        return float(s)
    except:
        return np.nan

df["monthly_charges_clean"] = df["monthly_charges"].apply(clean_money)
df["total_charges_clean"] = df["total_charges"].apply(clean_money)

# 标记缺失
df["missing_monthly_charges"] = df["monthly_charges_clean"].isna()
df["missing_total_charges"] = df["total_charges_clean"].isna()

# 去重
df = df.drop_duplicates(subset="customer_id",keep="first")

# 衍生变量
df["is_churn"] = df["churn"].map({"Yes":1,"No":0})
df["total_charges_calc"] = df["tenure"] * df["monthly_charges_clean"]
df["is_internet"] = df["internet_service"].apply(lambda x:0 if str(x).upper()=="NO" else 1)
df["has_tech_support"] = df["tech_support"].apply(lambda x:1 if str(x).upper()=="YES" else 0)

# 保存清洗后数据
df.to_csv("telecom_churn_cleaned.csv",index=False,encoding="utf8")
print("清洗完成，保存 telecom_churn_cleaned.csv")

# ==========3 EDA 基础统计 ==========
churn_rate = df["is_churn"].mean()
print(f"整体流失率：{churn_rate:.2%}")

# 分组流失率示例
print("\n====不同合约流失率====")
print(df.groupby("contract_clean")["is_churn"].agg(["count","mean"]))

# ==========4 卡方检验 分类特征 ==========
cat_features = ["contract_clean","internet_service","partner","dependents"]
for feat in cat_features:
    ct = pd.crosstab(df[feat],df["is_churn"])
    chi2,p,_,_ = chi2_contingency(ct)
    print(f"\n{feat} 卡方检验 p值={p:.4f}")

# 数值特征t检验
churn_yes = df.loc[df["is_churn"]==1,"tenure"].dropna()
churn_no = df.loc[df["is_churn"]==0,"tenure"].dropna()
t_stat,p_t = stats.ttest_ind(churn_yes,churn_no)
print(f"\ntenure t检验 p={p_t:.4f}")

# ==========5 建模验证 ==========
feature_cols = ["tenure","monthly_charges_clean","is_internet","has_tech_support"]
df_model = df.dropna(subset=feature_cols+["is_churn"])
X = df_model[feature_cols]
y = df_model["is_churn"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# 逻辑回归
lr = LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)
lr_pred = lr.predict_proba(X_test)[:,1]
print(f"\n逻辑回归 AUC: {roc_auc_score(y_test,lr_pred):.3f}")

# 随机森林
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict_proba(X_test)[:,1]
print(f"随机森林 AUC: {roc_auc_score(y_test,rf_pred):.3f}")
