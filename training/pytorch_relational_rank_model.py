import argparse
import copy
import numpy as np
import random
import datetime
import glob
from time import time
import os
os.environ['KMP_WARNINGS'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from evaluator import evaluate

def weighted_mse_loss(inputs, target, weights):
    # print('return_ratio are of size', inputs.shape, ' and is', inputs)
    # print('ground_truth are of size', target.shape, ' and is', target)
    # print('mask are of size', weights.shape, ' and is', weights)
    out = inputs - target
    # print('out are of size', out.shape, ' and is', out)
    out = out * out
    # print('out are of size', out.shape, ' and is', out)
    return out.sum(0)

def get_pretrained_weights(model, device, directory="pretrained_model", get_any=False):
    latest_model = None
    prev_models = glob.glob(directory + '/*.pth')
    if prev_models:
        latest_model = max(prev_models, key=os.path.getctime)
    if (latest_model is not None):  
        print('loading model', latest_model)
        model.load_state_dict(torch.load(latest_model, map_location=device))  
        return model, True
    else:
        print('no model found. train a new one.')
        return model, False

seed = 123456789
np.random.seed(seed)
torch.manual_seed(seed)

class TorchReRaLSTM(torch.nn.Module):
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, params, device, steps, batch_size=None, 
                 flat=False, in_pro=False):
        
        super(TorchReRaLSTM, self).__init__()
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
        if self.relation_name == 'correlational':
            self.rel_encoding = load_corr_timestep(market_name=self.market_name, 
                                                   t=0).squeeze_().unsqueeze_(-1).requires_grad_(True)
            rel_shape = self.rel_encoding.shape
            self.rel_mask = torch.where(self.rel_encoding == 0, 
                                        torch.ones(rel_shape) * -1e9, 
                                        torch.zeros(rel_shape)).squeeze_()
        else:
            rname_tail = {'sector_industry': '_industry_relation.npy',
                          'wikidata': '_wiki_relation.npy'}

            self.rel_encoding, self.rel_mask = load_relation_data(
                os.path.join(self.data_path, '..', 'relation', self.relation_name,
                             self.market_name + rname_tail[self.relation_name])
            )
            self.rel_encoding = torch.from_numpy(self.rel_encoding).float()
            self.rel_mask = torch.from_numpy(self.rel_mask).float()
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)

        self.embedding = np.load(
            os.path.join(self.data_path, '..', 'pretrain', emb_fname))
            # print('embedding shape:', self.embedding.shape)

        self.params = copy.copy(params)
        self.steps = steps
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

        # self.gpu = gpu

        # random shuffling of the batch data
        self.device = device
        self.batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        np.random.shuffle(self.batch_offsets)
        
        self.lrelu = nn.LeakyReLU(0.2)
        self.rel_weightlayer = nn.Linear(self.rel_encoding.shape[-1], 1)
        self.head_weightlayer = nn.Linear(self.params['unit'], 1)
        self.tail_weightlayer = nn.Linear(self.params['unit'], 1)
        
        self.weight_maskedsoftmax = nn.Softmax(dim = 0)
        self.outputs_concatedlayer = nn.Linear(self.params['unit'] * 2, self.params['unit'])
        gain=nn.init.calculate_gain('leaky_relu', 0.2)
        # nn.init.xavier_uniform_(self.outputs_concatedlayer, gain)
        
        self.predictionlayer = nn.Linear(self.params['unit'] * 2, 1)
        # nn.init.xavier_uniform_(self.predictionlayer, gain)

        self.prev_relation = self.rel_encoding

        self.money_after_days = 0


    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        #gives the length of the sequence
        seq_len = self.params['seq']

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
    
    def forward(self, j, epoch = 1):

        # the ground truths, mask, features and base_price are placeholders of sizes [batchsize x 1], i.e. vectors
        ground_truth = torch.empty(self.batch_size, 1).to(device)
        mask = torch.empty(self.batch_size, 1).to(device)

        # feature is a matrix of size [batchsize x params's unit]
        feature = torch.empty(self.batch_size, self.params['unit']).to(device)
        base_price = torch.empty(self.batch_size,1).to(device)

        all_one = torch.ones([self.batch_size, 1], dtype=torch.float64).to(device)

        # getting the data
        if j < 756:
            emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(self.batch_offsets[j])
        else:
            emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(j)
        
        # feature is a matrix of size [batchsize x params's unit]
        feature = torch.tensor(emb_batch).to(device)
        
        # the ground truths, mask, features and base_price are placeholders of sizes [batchsize x 1], i.e. vectors
        mask = torch.tensor(mask_batch).to(device)
        ground_truth = torch.tensor(gt_batch).to(device)
        base_price = torch.tensor(price_batch).to(device)
      
        if self.relation_name == 'correlational':
            self.rel_encoding = load_corr_timestep(market_name=self.market_name, 
                                                   t=j).squeeze_().unsqueeze_(-1).requires_grad_(True)
            rel_shape = self.rel_encoding.shape
            self.rel_mask = torch.where(self.rel_encoding == 0, 
                                        torch.ones(rel_shape) * -1e9, 
                                        torch.zeros(rel_shape)).squeeze_()

        relation = self.rel_encoding.to(device)
        rel_mask = self.rel_mask.float().to(device)

        rel_weight = self.rel_weightlayer(relation).clamp(min = 0)
        # print('rel_weight:', rel_weight.shape)
        
        if self.inner_prod:
            # print('inner product weight')
            inner_weight = torch.matmul(feature, torch.transpose(feature, 0,1).float())
            weight = torch.mul(inner_weight, rel_weight[:, :, -1])
        else:
            # print('sum weight')
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
        # print('weight', weight.shape)
        
        #rel_mask and weight are added. then an activation function is applied
        weight_masked = self.weight_maskedsoftmax(torch.add(rel_mask, weight))
        # print('weight_masked:', weight_masked.shape)
        
        #outputs_proped is weight_masked * feature
        outputs_proped = torch.matmul(weight_masked, feature)
        # print('outputs_proped:', outputs_proped.shape)
        
        if self.flat:
            # print('one more hidden layer')
            torch.cat(feature, outputs_proped, dim = 0)
            # outputs_concatedlayer = nn.Linear(feature.shape[-1], self.params['unit'])
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
            F.relu(
                torch.mul(
                    torch.mul(pre_pw_dif, gt_pw_dif),
                    mask_pw
                )
            )
        )
        
        # loss is then the params/rank_loss (scalar) + reg_loss, which was 
        # the MSE of ground truth and prediction rr's
        loss = reg_loss + self.params['alpha'] * rank_loss
        
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
    parser.add_argument('-g', '--gpu', type=int, default=1, help='try gpu, fallback to cpu')
    parser.add_argument('-e', '--emb_file', type=str,
                        default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
                        help='fname for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry, wikidata, or correlational')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-sv', '--save_model', default=True)
    parser.add_argument('-sp', '--save_model_path', default='pretrained_model')
    parser.add_argument('-up', '--use_pretrain', default=1, help='searches save_model_path \
                        for pretrained weights and skips training.')
    parser.add_argument('-rw', '--rolling_window', default = None, help='rolling window size')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    # check if CUDA is available
    device = torch.device("cpu")
    if args.gpu == True:
        if torch.cuda.is_available():
            device = torch.device("cuda")
    print('** training on', device, '**')

    args.inner_prod = (args.inner_prod == 1)
    params = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('params:', params)
    
    # Construct our model by instantiating the class defined above
    model = TorchReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        params=params,
        device=device,
        steps=args.s, 
        batch_size=None, 
        in_pro=args.inner_prod
    )

    model.to(device)

    model_found = False
    # if args.use_pretrain == True:
    #     model, model_found = get_pretrained_weights(model, device, args.save_model_path)
    
    if not args.use_pretrain or not model_found: 
    
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        epochs = args.epochs
        steps = args.s
        valid_index = 756
        test_index = 1008
        trade_dates = 1245
        fea_dim = 5
        tickers_len = 1026

        train_range = valid_index - params['seq'] - steps + 1
        valid_range = test_index - params['seq'] - steps + 1
        test_range = trade_dates - params['seq'] - steps + 1

        for roll in range(1, 1 + 1):
            i = roll
            
            best_valid_pred = best_valid_gt = best_valid_mask = np.zeros(
            [tickers_len, test_index - valid_index] ,
            dtype=float
            )
            
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0

            start = time()
            for t in range(train_range):
                # Forward pass: Compute predicted y by passing x to the model
                cur_loss, cur_reg_loss, cur_rank_loss= model(j = t)

                # Compute and print loss
                if t % 100 == 0:
                    loss = cur_loss.item() / train_range
                    print('Train Epoch: {} ({:.0f}%) \t Training Loss: {:.6f} \t '.format(
                        i, 100. * (t/train_range), loss))
                
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

                # save the trained correlational weights for timestep t
                if args.rel_name == 'correlational':
                    corr_t = model.rel_encoding.clone().detach()
                    grad_t = model.rel_encoding.grad
                    corr_t -= (grad_t * params['lr'])
                    save_corr_timestep(data=corr_t, 
                                        market_name=model.market_name, 
                                        t=t)

            tra_loss = tra_loss.detach().cpu().numpy() / train_range
            tra_reg_loss = tra_reg_loss.detach().cpu().numpy() / train_range
            tra_rank_loss = tra_rank_loss.detach().cpu().numpy() / train_range

            print('Train Loss:', tra_loss.item(), 
                                    tra_reg_loss.item(), 
                                    tra_rank_loss.item())

            if args.save_model:
                path = args.save_model_path
                if not os.path.exists(path):
                    os.makedirs(path)
                str_date = str(datetime.date.today())
                save_file_path = os.path.join(path, 'model_' + str(epochs) + 'epochs_' 
                                                + model.market_name + '_'
                                                + model.relation_name + '_loss'
                                                + str(round(tra_loss.item(),2)) + '_'
                                                + str_date + '.pth')
                torch.save(model.state_dict(), save_file_path)

            print('training complete in', str(time() - start), 'seconds')


        val_loss = 0.0
        val_reg_loss = 0.0
        val_rank_loss = 0.0
        
        for cur_offset in range(train_range, valid_range):
            # Forward pass: Compute predicted y by passing x to the model
            cur_loss, cur_reg_loss, cur_rank_loss= model(j = cur_offset)

            val_loss += cur_loss
            val_reg_loss += cur_reg_loss
            val_rank_loss += cur_rank_loss

            if t % 100 == 0:
                loss = cur_loss.item() / (valid_range - train_range)
                print('Train Epoch: {} ({:.0f}%) \t Validation Loss: {:.6f} \t '.format(
                    i, 100. * (t/valid_range - train_range), loss))

            # cur_valid_pred[:, cur_offset - (train_range] = \
            #     copy.copy(cur_rr[:, 0])
            # cur_valid_gt[:, cur_offset - (train_range)] = \
            #     copy.copy(gt_batch[:, 0])
            # cur_valid_mask[:, cur_offset - (train_range)] = \
            #     copy.copy(mask_batch[:, 0])

        # cur_valid_pred = cur_valid_gt = cur_valid_mask = np.zeros(
        #     [tickers_len, test_index - valid_index],
        #     dtype=float
        # )

        print('Valid MSE:',
        val_loss.detach().numpy() / (valid_range - train_range),
        val_reg_loss.detach().numpy() / (valid_range - train_range),
        val_rank_loss.detach().numpy() / (valid_range - train_range))

        test_loss = 0.0
        test_reg_loss = 0.0
        test_rank_loss = 0.0

        for cur_offset in range(valid_range, test_range):
            cur_loss, cur_reg_loss, cur_rank_loss= model(j = cur_offset)

            if t % 100 == 0:
                loss = cur_loss.item() / (test_range - valid_range)
                print('Train Epoch: {} ({:.0f}%) \t Testing Loss: {:.6f} \t '.format(
                    i, 100. * (t/test_range - valid_range), loss))

            test_loss += cur_loss
            test_reg_loss += cur_reg_loss
            test_rank_loss += cur_rank_loss
        
        print('Test MSE:',
        test_loss.detach().numpy() / (test_range - valid_range),
        test_reg_loss.detach().numpy() / (test_range - valid_range),
        test_rank_loss.detach().numpy() / (test_range - valid_range))










    print('training complete')

# best_valid_pred = best_valid_gt = best_valid_mask = np.zeros(
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
