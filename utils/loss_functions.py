import torch


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.nn.functional.mse_loss(torch.bmm(trans, trans.transpose(2, 1)), I)
    return loss
