import copy
import numpy as np


from FL_utils import agg_func, knowledge_aggregation
from get_dataset import get_data
from NN_models import MLP_Mnist, CNNCifar
from model_train import LocalUpdate_KFL
from model_test import test_img_local_all



def server_heteroM_run(args):

    # user_data_samples = np.ones(args.num_users)

    # get dataset
    dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all = get_data(args)
    # shuffle the local datasets:
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])


    # get models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i < 20:
                    hidden_dim = 128
                elif i >= 20 and i < 40:
                    hidden_dim = 192
                elif i >= 40 and i < 60:
                    hidden_dim = 256
                elif i >= 60 and i < 80:
                    hidden_dim = 320
                else:
                    hidden_dim = 384
            else:
                hidden_dim = 256
            local_model = MLP_Mnist(dim_in=784, dim_hidden=hidden_dim, dim_out=10)

        elif args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i < 20:
                    hidden_dim = 128
                elif i >= 20 and i < 40:
                    hidden_dim = 192
                elif i >= 40 and i < 60:
                    hidden_dim = 256
                elif i >= 60 and i < 80:
                    hidden_dim = 320
                else:
                    hidden_dim = 384
            else:
                hidden_dim = 128
            local_model = CNNCifar(args, hidden_dim=hidden_dim)


        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)


    # ---------------------------training-------------------------------------------------
    accs = []
    loss_round = []
    global_knows = []

    total_loss_train = []
    empirical_loss_train = []
    knowledge_loss_train = []

    for iter in range(args.epochs + 1):
        # device sampling:
        m = max(int(args.frac * args.num_users), 1)  # number of selected users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # random select users

        local_knows = {}

        total_loss_trains = []
        empirical_loss_trains = []
        knowledge_loss_trains = []

        for ind, idx in enumerate(idxs_users):  # selected users
            local = LocalUpdate_KFL(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            # copy the local model:
            local_model = copy.deepcopy(local_model_list[idx])
            w, loss, knows = local.train(local_model, global_knows, lr=args.lr)
            agg_knows = agg_func(knows)

            # save the local model:
            local_model.load_state_dict(copy.deepcopy(w), strict=True)
            local_model_list[idx] = local_model

            # collect the local knowledge:
            local_knows[idx] = agg_knows

            # record loss:
            total_loss_trains.append(copy.deepcopy(loss['total']))
            empirical_loss_trains.append(copy.deepcopy(loss['1']))
            knowledge_loss_trains.append(copy.deepcopy(loss['2']))

        total_train_l = sum(total_loss_trains)/len(total_loss_trains)
        total_loss_train.append(total_train_l)
        empirical_loss_l = sum(empirical_loss_trains)/len(empirical_loss_trains)
        empirical_loss_train.append(empirical_loss_l)
        know_loss_l = sum(knowledge_loss_trains)/len(knowledge_loss_trains)
        knowledge_loss_train.append(know_loss_l)

        # aggregate the local knowledges:
        global_knows = knowledge_aggregation(local_knows)

        # test:
        acc_test, loss_test = test_img_local_all(local_model_list, args, dataset_test, dict_users_test, return_all=False)
        accs.append(acc_test)
        loss_round.append(loss_test)
        print('Round {:3d}, Train loss: {:.3f}, EM_Train loss: {:.3f}, pro_Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, total_train_l, empirical_loss_l, know_loss_l, loss_test, acc_test))

    return accs, loss_round, total_loss_train, empirical_loss_train, knowledge_loss_train


