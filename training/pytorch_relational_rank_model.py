import argparse
import copy
import numpy as np
import random
import datetime
import glob
import math
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

        self.num_tickers = len(self.tickers)
        print('#tickers selected:', self.num_tickers)

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

        # self.valid_index = 756
        # self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        # self.gpu = gpu

        # random shuffling of the batch data
        self.device = device
        # self.batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        # np.random.shuffle(self.batch_offsets)
        
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
    
    def forward(self, j):
        
        all_one = torch.ones([self.batch_size, 1], dtype=torch.float64).to(device)

        return_ratio, ground_truth, mask = self.predict(start_idx=j)
        
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

    def predict(self, start_idx, end_idx=None):
        '''
        Parameters:
        start_idx : int, timestep to start predictions at
        end_idx   : int, timestep to end predictions at. If None, 
                         then only return for 1 timestep at start_idx
        '''

        if end_idx is None:
            end_idx = start_idx + 1

        num_steps = end_idx - start_idx

        return_ratios = []
        ground_truths = []
        masks = []

        for j in range(start_idx, end_idx):
            # the ground truths, mask, features and base_price are placeholders of sizes [batchsize x 1], i.e. vectors
            ground_truth = torch.empty(self.batch_size, 1).to(device)
            mask = torch.empty(self.batch_size, 1).to(device)

            # feature is a matrix of size [batchsize x params's unit]
            feature = torch.empty(self.batch_size, self.params['unit']).to(device)
            base_price = torch.empty(self.batch_size,1).to(device)

            all_one = torch.ones([self.batch_size, 1], dtype=torch.float64).to(device)

            # # getting the data
            # if j < 756:
            #     emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(self.batch_offsets[j])
            # else:
            #     emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(j)

            emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(j)
            
            # feature is a matrix of size [batchsize x params's unit]
            feature = torch.tensor(emb_batch).to(device)
            
            # the ground truths, mask, features and base_price are placeholders of sizes [batchsize x 1], i.e. vectors
            mask = torch.tensor(mask_batch).to(device)
            ground_truth = torch.tensor(gt_batch).to(device)
            base_price = torch.tensor(price_batch).to(device)

            ground_truths.append(ground_truth)
            masks.append(mask)
          
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
            return_ratios.append(torch.div( (prediction - base_price), base_price))

        return_ratios = torch.cat(return_ratios, dim=1).float().to(device)
        ground_truths = torch.cat(ground_truths, dim=1).float().to(device)
        masks = torch.cat(masks, dim=1).float().to(device)

        return return_ratios, ground_truths, masks

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
    parser.add_argument('-trs', '--train_size', default=200, help='size of training window')
    parser.add_argument('-vas', '--val_size',   default=20,  help='size of validation window')
    parser.add_argument('-tes', '--test_size',  default=20,  help='size of testing window')
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
    if args.use_pretrain == True:
        model, model_found = get_pretrained_weights(model, device, args.save_model_path)
    
    tickers_len = len(model.tickers)

    epochs = args.epochs
    steps = args.s

    num_timesteps = 1215    
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    num_windows = math.floor(num_timesteps / train_size) # 5
    print('num_windows:', num_windows)

    start_idx = 0

    if not args.use_pretrain or not model_found: 

        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        print_freq = math.floor(train_size / 5)

        start = time()

        for window in range(1, num_windows + 1):

            # update indices for the rolling window
            valid_index = start_idx   + train_size # originally 756 
            test_index  = valid_index + val_size   # originally 1008
            trade_dates = test_index  + test_size  # originally 1215

            train_range = valid_index - params['seq'] - steps + 1
            valid_range = test_index  - params['seq'] - steps + 1
            test_range  = trade_dates - params['seq'] - steps + 1

            print('window:\t', window)
            print('start_idx:\t', start_idx)
            print('train_range:\t', train_range)
            print('valid_range:\t', valid_range)
            print('test_range:\t', test_range)

            ### TRAIN #################################################################
            for epoch in range(1, epochs + 1):
                
                best_valid_pred = best_valid_gt = best_valid_mask = np.zeros(
                [tickers_len, test_index - valid_index] ,
                dtype=float
                )
                
                tra_loss = 0.0
                tra_reg_loss = 0.0
                tra_rank_loss = 0.0

                for i, t in enumerate(range(start_idx, train_range - 1)):
                    # Forward pass: Compute predicted y by passing x to the model
                    cur_loss, cur_reg_loss, cur_rank_loss= model(j = t)

                    # Compute and print loss
                    if t % print_freq == 0:
                        loss = cur_loss.item() / train_size
                        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t Time: {:.2f}'.format(
                               epoch, 
                               t, 
                               train_range,
                               100. * (i/train_size), 
                               loss, 
                               (time() - start)))
                    
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
                tra_loss = tra_loss.detach().cpu().numpy() / train_size
                tra_reg_loss = tra_reg_loss.detach().cpu().numpy() / train_size
                tra_rank_loss = tra_rank_loss.detach().cpu().numpy() / train_size

                print('Train loss: {:.6f} \t reg_loss: {:.6f} \t rank_loss: {:.6f}'.format(
                    tra_loss.item(),
                    tra_reg_loss.item(),
                    tra_rank_loss.item()))

            if args.save_model:
                path = args.save_model_path
                if not os.path.exists(path):
                    os.makedirs(path)
                str_date = str(datetime.date.today())
                save_file_path = os.path.join(path, 'model_' + str(epochs) + 'epochs_roll' 
                                                + str(window) + '_'
                                                + model.market_name + '_'
                                                + model.relation_name + '_loss'
                                                + str(round(tra_loss.item(),2)) + '_'
                                                + str_date + '.pth')
                torch.save(model.state_dict(), save_file_path)

            print('training complete in', str(time() - start), 'seconds')

            ### VALIDATION ############################################################
            # Note: validation is done using the last tensor from training
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            
            for cur_offset in range(train_range, valid_range - 1):
                cur_loss, cur_reg_loss, cur_rank_loss= model(j = valid_index-1)

                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss

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

            val_loss = val_loss.detach().cpu().numpy() / val_size
            val_reg_loss = val_reg_loss.detach().cpu().numpy() / val_size
            val_rank_loss = val_rank_loss.detach().cpu().numpy() / val_size
            
            print('Valid loss: {:.6f} \t reg_loss: {:.6f} \t rank_loss: {:.6f}'.format(
                val_loss.item(),
                val_reg_loss.item(),
                val_rank_loss.item()))

            ### TEST ##################################################################
            # Note: validation is done using the last tensor from training
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0

            for cur_offset in range(valid_range, test_range - 1):
                cur_loss, cur_reg_loss, cur_rank_loss= model(j = valid_index - 1)

                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss

            test_loss = test_loss.detach().cpu().numpy() / test_size
            test_reg_loss = test_reg_loss.detach().cpu().numpy() / test_size
            test_rank_loss = test_rank_loss.detach().cpu().numpy() / test_size
            
            print('Test loss: {:.6f} \t reg_loss: {:.6f} \t rank_loss: {:.6f}'.format(
                test_loss.item(),
                test_reg_loss.item(),
                test_rank_loss.item()))

            # move the starting index up
            start_idx = train_range

    print('training complete')

    print('begin evaluation')

    start_idx = 0

    return_ratios = []
    ground_truths = []

    for window in range(1, num_windows + 1):

        # update indices for the rolling window
        valid_index = start_idx   + train_size
        test_index  = valid_index + val_size 
        trade_dates = min(test_index  + test_size, num_timesteps - 1) 

        train_range = valid_index - params['seq'] - steps + 1
        valid_range = test_index  - params['seq'] - steps + 1
        test_range  = trade_dates - params['seq'] - steps + 1

        print('window:\t', window)
        print('start_idx:\t', start_idx)
        print('train_range:\t', train_range)
        print('valid_range:\t', valid_range)
        print('test_range:\t', test_range)

        return_ratio, ground_truth, _ = model.predict(start_idx=test_index, end_idx=trade_dates)

        return_ratios.append(return_ratio)
        ground_truths.append(ground_truth)

        # move the starting index up
        start_idx = train_range

    return_ratios = torch.cat(return_ratios, dim=1).T
    ground_truths = torch.cat(ground_truths, dim=1).T

    print(return_ratios.shape, ground_truths.shape)
    # print(return_ratios, ground_truths)

    # calculate MSE
    MSE = F.mse_loss(return_ratios, ground_truths)
    print('MSE:', MSE.item())

    # calculate returns
    daily_investment = 100

    # our returns for dataset
    best_pred_gain, best_pred_idxs = torch.max(return_ratios, axis=1)
    tsteps = torch.arange(0, len(best_pred_idxs)).long()
    earn_gain = ground_truths[(tsteps, best_pred_idxs)]

    # best possible for dataset
    best_gt_gain = torch.max(ground_truths, axis=1)[0]

    # average for dataset
    mean_gt_gain = torch.mean(ground_truths, axis=1)[0]

    pred_return = torch.sum(torch.mul(daily_investment, best_pred_gain))
    earn_return = torch.sum(torch.mul(daily_investment, earn_gain))
    best_return = torch.sum(torch.mul(daily_investment, best_gt_gain))
    mean_return = torch.sum(torch.mul(daily_investment, mean_gt_gain))

    print('pred_return:', pred_return.item())
    print('earn_return:', earn_return.item())
    print('best_return:', best_return.item())
    print('mean_return:', mean_return.item())
