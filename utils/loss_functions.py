import torch


def feature_transform_regularizer(trans):
    d = trans.size()[1]    # (batch_size, k, k)
    I = torch.eye(d)[None, :, :]  # torch.eye(d)是一个二维矩阵，[None,:,:]增加了一个维度，batch维度为1
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.nn.functional.mse_loss(torch.bmm(trans, trans.transpose(2, 1)), I)
    return loss
