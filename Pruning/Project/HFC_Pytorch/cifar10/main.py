import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from collections import OrderedDict
import numpy as np
from scipy import signal
from time import time
from model import WideResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from attack import FastGradientSignUntargeted
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate
import argparse


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i, j] = 1
            else:
                mask[i, j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:, :, i, j] = signal.convolve2d(data[:, :, i, j], mask, boundary='symm', mode='same')
    return result


def load_model(model, file_name, args):
    state_dict = torch.load(file_name, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, data in state_dict.items():
        if args.rho > 0:
            if k == 'conv1.weight':
                # if k.find('weight') != -1:
                #     if k.find('conv1.') != -1:
                data = generateSmoothKernel(data.numpy(), args.rho)
                data = torch.from_numpy(data)
        new_state_dict[k] = data
    model.load_state_dict(new_state_dict)


def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 150], gamma=0.1)
        _iter = 0

        begin_time = time()

        for epoch in range(1, args.max_epoch + 1):
            scheduler.step()
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # When training, the adversarial example is created from a random
                    # close point to the original data point. If in evaluation mode,
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    t1 = time()

                    if adv_train:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:

                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time()

                    print('%.3f' % (t2 - t1))

                    print('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (epoch, _iter, time() - begin_time, loss.item()))

                    print('standard acc: %.3f %%, robustness acc: %.3f %%' % (std_acc, adv_acc))

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                # if _iter % args.n_store_image_step == 0:
                #     tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0),
                #                         os.path.join(args.log_folder, 'images_%d.jpg' % _iter),
                #                         nrow=16)

                _iter += 1

            file_name = os.path.join(args.model_folder, 'checkpoint.pth')
            torch.save(model.state_dict(), file_name)

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                print('\n'+'='*20 +' evaluation at epoch: %d iteration: %d '%(epoch, _iter) \
                    +'='*20)
                print('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (va_acc, va_adv_acc, t2 - t1))
                print('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

    def test(self, model, loader, adv_test=False):
        # adv_test is False, return adv_acc as -1

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, pred, 'mean', False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num, total_adv_acc / num


def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    #return
    # model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
    print('create model')
    model = ResNet34()
    print('done...')
    # file_name = os.path.join(args.model_folder, 'checkpoint.pth')
    # torch.save(model.state_dict(), file_name)
    # load_model(model, file_name, args)

    print('create attack')
    attack = FastGradientSignUntargeted(model,
                                        args.epsilon,
                                        args.alpha,
                                        min_val=0,
                                        max_val=1,
                                        max_iters=args.k,
                                        _type=args.perturbation_type)
    print('done...')

    print('create trainer')
    trainer = Trainer(args, logger, attack)
    print('done...')

    if args.todo == 'train':
        print('start train')
        if torch.cuda.is_available():
            model.cuda()
        transform_train = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='constant', value=0).squeeze()),
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(32),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])

        print('get dataset')
        print('dataset_root: ', args.data_root)
        tr_dataset = tv.datasets.CIFAR10(args.data_root, train=True, transform=transform_train, download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root, train=False, transform=tv.transforms.ToTensor(), download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'valid':
        load_model(model, args.load_checkpoint, args)

        if torch.cuda.is_available():
            model.cuda()
        te_dataset = tv.datasets.CIFAR10(args.data_root, train=False, transform=tv.transforms.ToTensor(), download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        test_acc, adv_acc = trainer.test(model, te_loader, adv_test=False)
        print('Test accuracy is %.3f' % test_acc)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Summarization')
    parser.add_argument('--todo', choices=['train', 'valid'], default='train', help='what behavior want to do: train | valid')
    parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
    parser.add_argument('--data_root', default='./cifar10/data', help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./cifar10/model/default/model.pth')
    parser.add_argument('--affix', default='default', help='the affix for the save folder')

    # parameters for generating adversarial examples
    parser.add_argument('--epsilon',
                        '-e',
                        type=float,
                        default=0.0157,
                        help='maximum perturbation of adversaries (4/255=0.0157)')
    parser.add_argument('--alpha',
                        '-a',
                        type=float,
                        default=0.00784,
                        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', '-k', type=int, default=10, help='maximum iteration when generating adversarial examples')

    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=200, help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, help='the parameter of l2 restriction for weights')

    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--n_eval_step', type=int, default=10, help='number of iteration per one evaluation')
    parser.add_argument('--n_store_image_step', type=int, default=4000, help='number of iteration to save adversaries')
    parser.add_argument('--perturbation_type',
                        '-p',
                        choices=['linf', 'l2'],
                        default='linf',
                        help='the type of the perturbation (linf or l2)')

    parser.add_argument('-r',
                        '--rho',
                        type=float,
                        default=1,
                        help='the rho of smoothing convolutional kernel, rho=0 for not using the method')

    parser.add_argument('--adv_train', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)

    #print(os.getcwd())
