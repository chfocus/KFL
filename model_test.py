import copy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label


def test_img_local(net_g, dataset, args, idxs=None):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=False)

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs, _ = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    count = len(data_loader.dataset)

    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return accuracy, test_loss


def test_img_local_all(net_list, args, dataset_test, dict_users_test, return_all=False):
    tot = 0
    num_idxxs = args.num_users # user number
    acc_test_local = np.zeros(num_idxxs)
    acc_locals = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):  # for users,
        net_local = copy.deepcopy(net_list[idx])  # locla model
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, idxs=dict_users_test[idx])
        tot += len(dict_users_test[idx])

        acc_locals[idx] = a / 100

        acc_test_local[idx] = a * len(dict_users_test[idx])
        loss_test_local[idx] = b * len(dict_users_test[idx])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot





