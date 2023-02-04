import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label




class LocalUpdate_KFL(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.knowledge_loss_func = nn.MSELoss()

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs

    def train(self, net, global_knows, lr=0.05):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': 0.0001},
                                     {'params': bias_p, 'weight_decay': 0}], lr=lr, momentum=0.5)

        local_eps = self.args.local_ep
        # epoch_loss = []
        epoch_loss = {'total': [], '1': [], '2': []}

        agg_knows_label = {}
        for iter in range(local_eps):
            # batch_loss = []
            batch_loss = {'total': [], '1': [], '2': []}
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.train()

                optimizer.zero_grad()
                log_probs, knows = net(images)
                empirical_loss = self.loss_func(log_probs, labels)

                # compute knowledge loss:
                knowledge_loss = 0 * empirical_loss
                if len(global_knows) == 0:
                    knowledge_loss = 0 * empirical_loss
                else:
                    know_temp = copy.deepcopy(knows.data)
                    i = 0
                    for l in labels:
                        if l.item() in global_knows.keys():
                            know_temp[i, :] = global_knows[l.item()][0].data
                        i += 1

                    knowledge_loss = self.knowledge_loss_func(know_temp, knows)

                loss = empirical_loss + self.args.ld * knowledge_loss
                loss.backward()
                optimizer.step()

                # record the loss:
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(empirical_loss.item())
                batch_loss['2'].append(knowledge_loss.item())

                # collect knowledges:
                if iter == local_eps - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_knows_label:
                            agg_knows_label[labels[i].item()].append(knows[i, :])
                        else:
                            agg_knows_label[labels[i].item()] = [knows[i, :]]

            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])


        return net.state_dict(), epoch_loss, agg_knows_label
