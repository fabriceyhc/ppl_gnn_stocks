

    # best_valid_pred = np.zeros(
#     [len(self.tickers), self.test_index - self.valid_index],
#     dtype=float
# )
# best_valid_gt = np.zeros(
#     [len(self.tickers), self.test_index - self.valid_index],
#     dtype=float
# )
# best_valid_mask = np.zeros(
#     [len(self.tickers), self.test_index - self.valid_index],
#     dtype=float
# )
# best_test_pred = np.zeros(
#     [len(self.tickers), self.trade_dates - self.parameters['seq'] -
#      self.test_index - self.steps + 1], dtype=float
# )
# best_test_gt = np.zeros(
#     [len(self.tickers), self.trade_dates - self.parameters['seq'] -
#      self.test_index - self.steps + 1], dtype=float
# )
# best_test_mask = np.zeros(
#     [len(self.tickers), self.trade_dates - self.parameters['seq'] -
#      self.test_index - self.steps + 1], dtype=float
# )
# best_valid_perf = {
#     'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
# }
# best_test_perf = {
#     'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
# }
# best_valid_loss = np.inf

import argparse
import copy
import numpy as np
import random
from time import time
import os
os.environ['KMP_WARNINGS'] = '0'
# import psutil

import torch
import torch.nn as nn
import torch.optim as optim

# check if CUDA is available
device = torch.device("cpu")
use_cuda = False
if torch.cuda.is_available():
    print('CUDA is available!')
    device = torch.device("cuda")
    use_cuda = True
else:
    print('CUDA is not available.')

from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate


def weighted_mse_loss(inputs, target, weights):
#     print('return_ratio are of size', inputs.shape, ' and is', inputs)
#     print('ground_truth are of size', target.shape, ' and is', target)
#     print('mask are of size', weights.shape, ' and is', weights)
    out = inputs - target
#     print('out are of size', out.shape, ' and is', out)
    out = out * out
#     print('out are of size', out.shape, ' and is', out)
    return out.sum(0)

relu = nn.LeakyReLU(0.2)
seed = 123456789
np.random.seed(seed)

# tf.set_random_seed(seed)
torch.manual_seed(seed)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameter, steps=1, epochs=50, batch_size=None, 
                 flat=False, gpu=False, in_pro=False):
        
        super(TwoLayerNet, self).__init__()
        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name

        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)

        print('#tickers selected:', len(self.tickers))

        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        # relation data
        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}

        self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(self.data_path, '..', 'relation', self.relation_name,
                         self.market_name + rname_tail[self.relation_name])
        )
#         print('relation encoding shape:', self.rel_encoding.shape)
#         print('relation mask shape:', self.rel_mask.shape)

        self.embedding = np.load(
            os.path.join(self.data_path, '..', 'pretrain', emb_fname))
#         print('embedding shape:', self.embedding.shape)

        self.parameter = copy.copy(parameter)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu
        self.lrelu = nn.LeakyReLU(0.2)
        self.batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        np.random.shuffle(self.batch_offsets)
        
        self.rel_weightlayer = nn.Linear(self.rel_encoding.shape[-1], 1)
        self.head_weightlayer = nn.Linear(self.parameter['unit'], 1)
        self.tail_weightlayer = nn.Linear(self.parameter['unit'], 1)
        
        self.weight_maskedsoftmax = nn.Softmax(dim = 0)
        self.outputs_concatedlayer = nn.Linear(self.parameter['unit'] * 2, self.parameter['unit'])
        gain=nn.init.calculate_gain('leaky_relu', 0.2)
#         nn.init.xavier_uniform_(self.outputs_concatedlayer, gain)
        
        self.predictionlayer = nn.Linear(self.parameter['unit'] * 2, 1)
#         nn.init.xavier_uniform_(self.predictionlayer, gain)


    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        #gives the length of the sequence
        seq_len = self.parameter['seq']

        #mask_batch stores the mask_data from offset to offset + sequencee length + steps
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]

        #goes through each row and finds the smallest element in it
        mask_batch = np.min(mask_batch, axis=1)

        """
        returns:
        1. embedding's matrix associated with the offset
        2. make the mask_batch into a tensor of dimensions [mask_batch x 1]
        3. price_data with the same expansion o dimensions to make a tensor
        4. ground_truth data with the same expansion
        """
        return self.embedding[:, offset, :], np.expand_dims(mask_batch, axis=1), np.expand_dims(
                    self.price_data[:, offset + seq_len - 1], axis=1
                ), \
                np.expand_dims(
                    self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
                )
    
    def forward(self, j):
        if self.gpu == True:
            cuda = torch.device('cuda')
        else:
            device_name = 'cpu'
        if j == 0: print('device name:', device_name)

        # the ground truths, mask, features and base_price are placeholders of sizes [batchsize x 1], i.e. vectors
        ground_truth = torch.empty(self.batch_size, 1).to(device)
        mask = torch.empty(self.batch_size, 1).to(device)

        # feature is a matrix of size [batchsize x parameters's unit]
        feature = torch.empty(self.batch_size, self.parameter['unit']).to(device)
        base_price = torch.empty(self.batch_size,1).to(device)

        all_one = torch.ones([self.batch_size, 1], dtype=torch.float64).to(device)

        #Getting the data
        
        emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(self.batch_offsets[j])
        
        # feature is a matrix of size [batchsize x parameters's unit]
        feature = torch.tensor(emb_batch).to(device)
        
        # the ground truths, mask, features and base_price are placeholders of sizes [batchsize x 1], i.e. vectors
        mask = torch.tensor(mask_batch).to(device)
        ground_truth = torch.tensor(gt_batch).to(device)
        base_price = torch.tensor(price_batch).to(device)

        
        relation = torch.FloatTensor(self.rel_encoding).to(device)
        rel_mask = torch.FloatTensor(self.rel_mask).to(device)

        rel_weight = self.rel_weightlayer(relation).clamp(min = 0)

        
        if self.inner_prod:
#             print('inner product weight')
            inner_weight = torch.matmul(feature, torch.transpose(feature, 0,1).float())
            weight = torch.mul(inner_weight, rel_weight[:, :, -1])
        else:
#             print('sum weight')
            head_weight = self.head_weightlayer(feature).clamp(min = 0)
            tail_weight = self.tail_weightlayer(feature).clamp(min = 0)
            
            """
            weight is elementwise sum of
                (sum of (head_weight * (vector of 1's) T + vector of 1's * (tail_weight)T )
                + rel_weight[:, :, -1]) )
            """
            weight = torch.add(
                torch.add(
                    torch.matmul(head_weight, torch.transpose(all_one, 0,1).float()),
                    torch.matmul(all_one.float(), torch.transpose(tail_weight, 0,1).float())
                ), rel_weight[:, :, -1]
            )
        
        
        #rel_mask and weight are added. then an activation function is applied
        weight_masked = self.weight_maskedsoftmax(torch.add(rel_mask, weight))
        
        #outputs_proped is weight_masked * feature
        outputs_proped = torch.matmul(weight_masked, feature)
        
        if self.flat:
            print('one more hidden layer')
            torch.cat(feature, outputs_proped, dim = 0)
            # outputs_concatedlayer = nn.Linear(feature.shape[-1], self.parameter['unit'])
            outputs_concated =  self.lrelu(self.outputs_concatedlayer(torch.cat((feature, outputs_proped), 1)))
            
        else:
            outputs_concated = torch.cat((feature, outputs_proped), 1)
        
        prediction = self.lrelu(self.predictionlayer(outputs_concated))

        #return ratio = (prediction - price)/price
        return_ratio = torch.div( (prediction - base_price), base_price)
        
        #reg_loss = MSE(ground_truth - return_ratio)
        reg_loss = weighted_mse_loss(inputs = return_ratio, target = ground_truth, weights = mask)

        #prediction_pw_difference = return_ratio*(vector of 1's)T - vector of 1's * (return_ratio)T
        pre_pw_dif = torch.matmul(return_ratio, torch.transpose(all_one, 0, 1).float())
        - torch.matmul(all_one.float(), torch.transpose(return_ratio, 0, 1).float())

        #groundtruth_pw_difference = ground_truth*(vector of 1's)T - vector of 1's * (ground_truth)T (outer products)
        gt_pw_dif = torch.matmul(all_one.float(), torch.transpose(return_ratio, 0, 1))
        - torch.matmul(return_ratio, torch.transpose(all_one, 0, 1).float())

        #mask_pw = mask * (mask)T = [batch_size x batch_size] (outer product)
        mask_pw = torch.matmul(mask, torch.transpose(mask, 0, 1))

        #rank_loss = ReLU( pre_pw_dif * gt_pw_dif * mask_pw ) and then calculates the mean of elements
        rank_loss = torch.mean(
            relu(
                torch.mul(
                    torch.mul(pre_pw_dif, gt_pw_dif),
                    mask_pw
                )
            )
        )
        
        # loss is then the parameters/rank_loss (scalar) + reg_loss, which was 
        # the MSE of ground truth and prediction rr's
        loss = reg_loss + self.parameter['alpha'] * rank_loss
        
        return loss, reg_loss, rank_loss
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')

    parser.add_argument('-e', '--emb_file', type=str,
                        default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
                        help='fname for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)
    parameter = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameter)

    # parameter = {'seq': int(16), 'unit': int(64), 'lr': float(0.01),
    #             'alpha': float(0.1)}
    
    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameter=parameter,
        steps=1, epochs=50, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod
    )

    model.to(device)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=parameter['lr'])
    epochs = 4
    steps = 1
    valid_index = 756

    T = valid_index - parameter['seq'] - steps + 1

    for i in range(1, epochs + 1):
        t1 = time()
        # random shuffling of the batch data
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        
        for t in range(T):
            # Forward pass: Compute predicted y by passing x to the model
            cur_loss, cur_reg_loss, cur_rank_loss= model(j = t)

            # Compute and print loss
            if t % 100 == 0:
                loss = cur_loss.item() / T
                print('Train Epoch: {} ({:.0f}%) \t Loss: {:.6f} \t '.format(
                    i, 100. * (t/T), loss))
            
            tra_loss += cur_loss
            tra_reg_loss += cur_reg_loss
            tra_rank_loss += cur_rank_loss
            
    #         print('Train Loss:',
    #         tra_loss / (valid_index - parameter['seq'] - steps + 1),
    #         tra_reg_loss / (valid_index - parameter['seq'] - steps + 1),
    #         tra_rank_loss / (valid_index - parameter['seq'] - steps + 1))
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
        print('Train Loss:',
        tra_loss.detach().numpy() / (T),
        tra_reg_loss.detach().numpy() / (T),
        tra_rank_loss.detach().numpy() / (T) )


    print('training complete')