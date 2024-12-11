import torch

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    计算两个点云之间的F分数，以及对应的阈值。
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    #Precision (精确率)：目标点云中与预测点云的最近邻距离小于阈值的点所占的比例。
    #Recall (召回率)：预测点云中与目标点云的最近邻距离小于阈值的点所占的比例。

    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    #在这个 depo 中，dist1 和 dist2 是平方点云欧几里得距离，因此您应该相应地调整阈值。
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    #计算精确率和召回率
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)#计算F-score
    fscore[torch.isnan(fscore)] = 0#将nan值替换为0
    return fscore, precision_1, precision_2#返回F-score，精确率和召回率

