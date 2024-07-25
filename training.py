import sys, os
import torch
import torch.nn as nn
from model import DTANet
from utils import *
from torch_geometric import loader


def train(model, device, train_loader, optimizer, loss_fn, epoch, LOG_INTERVAL):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        data_pack = [data.x, data.edge_index, data.batch, data.target]
        edge_weight = torch.ones(data.edge_index.size(1)).to(device)
        data_pack.append(edge_weight)

        output = model(data_pack)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            data_pack = [data.x, data.edge_index, data.batch, data.target]
            edge_weight = torch.ones(data.edge_index.size(1)).to(device)
            data_pack.append(edge_weight)
            output = model(data_pack)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def main(dataset_index, cuda_index):
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1000

    datasets = [['davis','kiba'][dataset_index]]
    modeling = DTANet
    model_st = 'DTANet'

    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = "cuda:" + str(cuda_index)
    print('cuda_name:', cuda_name)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main part: iterate over different datasets
    for dataset in datasets:
        print('\nrunning on ', model_st + '_' + dataset )
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='data/', dataset=dataset+'_train')
            test_data = TestbedDataset(root='data/', dataset=dataset+'_test')
            
            # make data PyTorch mini-batch processing ready
            train_loader = loader.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
            test_loader = loader.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            model_file_path = 'final results/model_' + model_st + '_' + dataset +  '.model'
            result_file_path = 'final results/result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, loss_fn, epoch+1, LOG_INTERVAL)
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P), ci(G,P)]
                if ret[1]<best_mse:
                    torch.save(model.state_dict(), model_file_path)

                    with open(result_file_path,'w') as f:
                        f.write(','.join(map(str, ['rmse', 'mse', 'pearson', 'spearman', 'ci'])))
                        f.write('\n')
                        f.write(','.join(map(str, ret)))

                    best_epoch = epoch+1
                    best_mse = ret[1]
                    best_ci = ret[-1]
                    print('rmse improved at epoch ', best_epoch, '| best_mse, best_ci:', best_mse, best_ci, model_st, dataset)
                else:
                    print(ret[1],'No improvement since epoch ', best_epoch, '| best_mse, best_ci:', best_mse, best_ci, model_st, dataset)


if __name__ == '__main__':
    dataset_index = 0
    cuda_index = 0
    main(dataset_index, cuda_index)