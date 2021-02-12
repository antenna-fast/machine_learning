
# 准确率
# 预测正确的结果所占的比例  TP+TN/TP+TN+FP+FN
def get_accuracy(TP, FP, TN, FN):
    return (TP + TN) / (TP + TN + FP + FN)


# 精确率
# 所有被识别为正类别的样本中，真正为正样本的比例  TP/TP+FP
def get_precision(TP, FP):
    return TP / (TP + FP)


# 召回率
# 所有正样本中，被正确识别为正样本的比例  TP/TP+FN
def get_recall(TP, FN):
    return TP / (TP + FN)

