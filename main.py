from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scripts.class_balancing import balance_classes
from scripts.data_preprocessing import load_data, preprocess_data, process_time_data_scale
from scripts.correlation_analysis import calculate_spearman_correlation, plot_correlation_matrix
from scripts.hyperparameter_tuning import tune_random_forest_rs, tune_adaboost_rs, tune_knn_rs
from scripts.scaling_pca import scale_data, apply_pca
from scripts.evaluation import evaluate_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Load and preprocess data
data = load_data('data/training_data_fall2024.csv')
data = preprocess_data(data)

numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data = numeric_data.drop(columns=['precip'])

spearman_corr, p_values, high_corr_df = calculate_spearman_correlation(numeric_data, threshold=0.7)
plot_correlation_matrix(spearman_corr)
# Print highly correlated pairs
print(f"high_corr_df = \n{high_corr_df}")
"""这里直接不PCA数据，对月份数据进行三角函数化，保留其周期性，同时对0-1标签数据不正则化"""
# numeric_data_new = data.select_dtypes(include=['float64', 'int64'])
numerical_new = process_time_data_scale(numeric_data)

# Split dataset
mapping = {'high_bike_demand': 1, 'low_bike_demand': 0}
y = data['increase_stock'].replace(mapping).to_numpy().reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(numerical_new, y, test_size=0.2, random_state=42,
                                                    )

"""Choose whether we need to balance the dataset"""
# X_train_bal, y_train_bal = balance_classes(X_train, y_train)

X_train_bal, y_train_bal = X_train, y_train

# """试图使用LDA正则化，甚至降低了精度的作用"""
# lda = LinearDiscriminantAnalysis()
# X_train_bal_lda = lda.fit_transform(X_train_bal, y_train_bal.ravel())
# X_test_lda = lda.transform(X_test)


# Hyperparameter tuning for Random Forest
print("Tuning Random Forest...")
best_rf = tune_random_forest_rs(X_train_bal, y_train_bal)
# Evaluate the tuned Random Forest model
accuracy, f1, report = evaluate_model(best_rf, X_test, y_test)
print(f"Tuned Random Forest Accuracy: {accuracy:.2f}")
print(f"Tuned Random Forest F1 Score: {f1:.2f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(y_test, best_rf.predict(X_test)))
print(f"\n\n\n\n")

# print("Tuning Adaboost...")
# best_boot = tune_adaboost_rs(X_train_bal, y_train_bal)
# accuracy, f1, report = evaluate_model(best_boot, X_test, y_test)
# print(f"Tuned Adaboost Accuracy: {accuracy:.2f}")
# print(f"Tuned Adaboost F1 Score: {f1:.2f}")
# print("Classification Report:\n", report)
# print("Confusion Matrix:\n", confusion_matrix(y_test, best_boot.predict(X_test)))
# print(f"\n\n\n\n")
#
# print("Tuning KNN...")
# best_knn = tune_knn_rs(X_train_bal, y_train_bal)
# accuracy, f1, report = evaluate_model(best_knn, X_test, y_test)
# print(f"Tuned KNN Accuracy: {accuracy:.2f}")
# print(f"Tuned KNN F1 Score: {f1:.2f}")
# print("Classification Report:\n", report)
# print("Confusion Matrix:\n", confusion_matrix(y_test, best_knn.predict(X_test)))
# print(f"\n\n\n\n")

# print("Tuning Ridge...")
# best_ridge = tune_ridge_rs(X_train_bal, y_train_bal)
# accuracy, f1, report, conf_matrix = evaluate_model_linear(best_ridge, X_test, y_test)
# print(f"Tuned Ridge Accuracy: {accuracy:.2f}")
# print(f"Tuned Ridge F1 Score: {f1:.2f}")
# print("Classification Report:\n", report)
# print("Confusion Matrix:\n", confusion_matrix(y_test, best_ridge.predict(X_test)))
# print(f"\n\n\n\n")
# """训练次数1：（数据未平衡）对随机森林来说，数据有没有平衡是无所谓的，模型训练当中权重被设置成了balanced"""
# n_estimators_list = [50, 100, 200, 300, 400, 500, 600]
# max_depth_list = [5, 10, 15, 20, 25, 30, None]
#
# """训练次数2：（数据未平衡）参数超越500左右之后就对召回率和f1分数的提升产生递减，故寻找参数大致放在了100到500, 缩小间隔计算，同时观察图知道，现在丢弃
# 18之后的树深度"""
# n_estimators_list = [25 * i for i in range(4, 26, 2)]
# max_depth_list = [i for i in range(5, 16)]
#
# """训练次数3：（数据未平衡）继续看训练次数2的图可以发现，高f1值集中在，50-300树数量和10-20左右的树深度达到最高值，所以继续测试"""
# # n_estimators_list = [25 * i for i in range(2, 13, 1)]
# # max_depth_list = [i for i in range(10, 22, 2)]
#
# """训练次数4：（数据未平衡）"""
# # 第三次测试没有显著结论，为了搜寻参数空间，现在利用随机搜寻大致找出参数区间
#
# # n_estimators_list = [25 * i for i in range(2, 13, 1)]
# # max_depth_list = [i for i in range(10, 22, 2)]
# 超参数应该在训练集上进行调整，随后去比较测试集的情况，注意K折线方法，可以在进入测试集之前就估算好性能
# f1_scores, recall_scores = evaluate_parameters_rf(X_train_bal, y_train_bal, n_estimators_list, max_depth_list)
#
# # 绘制F1分数热力图
# """绘图参考文献：https://seaborn.pydata.org/generated/seaborn.heatmap.html"""
# max_depth_labels = [d if d is not None else 'None' for d in max_depth_list]
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(f1_scores, annot=True, fmt=".3f", xticklabels=n_estimators_list, yticklabels=max_depth_labels,
#             cmap="YlGnBu")
# plt.title("F1 Score Heatmap")
# plt.xlabel("n_estimators")
# plt.ylabel("max_depth")
# plt.show()
#
# # 绘制召回率热力图
# plt.figure(figsize=(10, 6))
# sns.heatmap(recall_scores, annot=True, fmt=".3f", xticklabels=n_estimators_list, yticklabels=max_depth_labels,
#             cmap="YlGnBu")
# plt.title("Recall Score Heatmap")
# plt.xlabel("n_estimators")
# plt.ylabel("max_depth")
# plt.show()

