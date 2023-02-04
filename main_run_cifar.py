import argparse
import numpy as np
import torch
import random
import os

from server_update import server_heteroM_run





def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='KFL', help="algorithms")
    parser.add_argument('--mode', type=str, default='task_heter', help="mode, data_heter, model_heter")
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    parser.add_argument('--ld', type=float, default=1, help="weight of knowledge loss")

    args = parser.parse_args()
    return args




def main():
    args = args_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set seed:
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    accs, loss, total_loss_train, empirical_loss_train, know_loss_train = server_heteroM_run(args)



if __name__ == '__main__':
    main()