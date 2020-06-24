import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import argparse
import os
from sklearn.metrics import accuracy_score
import sklearn.linear_model as models
from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--episode", type=int, default=500000)
parser.add_argument("-t", "--test_episode", type=int, default=2000)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

class AttributeNetwork(nn.Module):
    """docstring for AttributeNetwork"""

    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


def main():
    # step 1: init dataset
    data_set = 'CUB'       # AwA1, AwA2, CUB, aPY are alternative

    if data_set == 'AwA1':
        total_class = 50
        g_rate = 10000
        dataroot = './data/'
        dataset = 'AwA1_data'
        image_embedding = 'res101'
        class_embedding = 'original_att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['att'].T

        attribute_network = AttributeNetwork(85, 1600, 2048)
        relation_network = RelationNetwork(4096, 400)

        attribute_network.cuda(GPU)
        relation_network.cuda(GPU)

    elif data_set == 'AwA2':
        total_class = 50
        g_rate = 10000
        dataroot = './data'
        dataset = 'AwA2_data'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T

        attribute_network = AttributeNetwork(85, 1600, 2048)
        relation_network = RelationNetwork(4096, 400)

        attribute_network.cuda(GPU)
        relation_network.cuda(GPU)


    elif data_set == 'CUB':
        total_class = 200
        g_rate = 2500
        dataroot = './data'
        dataset = 'CUB1_data'
        image_embedding = 'res101'
        class_embedding = 'original_att_splits'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['att'].T

        attribute_network = AttributeNetwork(312, 1200, 2048)
        relation_network = RelationNetwork(4096, 1200)

        attribute_network.cuda(GPU)
        relation_network.cuda(GPU)

    elif data_set == 'aPY':
        total_class = 32
        g_rate = 5000
        dataroot = './data/'
        dataset = 'APY_data'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T

        attribute_network = AttributeNetwork(64, 1200, 2048)
        relation_network = RelationNetwork(4096, 400)

        attribute_network.cuda(GPU)
        relation_network.cuda(GPU)

    else:
        print('Wrong dataset!')

    if os.path.exists("./GRmodels/" + data_set + "_RN_R15_attribute_network.pkl"):
        attribute_network.load_state_dict(
            torch.load("./GRmodels/" + data_set + "_RN_R15_attribute_network.pkl"))
        print("load attribute network success")

    if os.path.exists("./GRmodels/" + data_set + "_RN_R15_relation_network.pkl"):
        relation_network.load_state_dict(
            torch.load("./GRmodels/" + data_set + "_RN_R15_relation_network.pkl"))
        print("load relation network success")

    x = feature[trainval_loc]  # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label]  # train attributes

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int)  # test_label
    x_test_seen = feature[test_seen_loc]  # test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
    train_id = np.unique(train_label)
    test_id = np.unique(test_label)  # test_id
    att_pro = attribute[test_id]  # test_attribute
    att_train = attribute[train_id]
    train_num = att_train.shape[0]
    test_num = att_pro.shape[0]

    class_train_x = []
    mu_train_x = np.zeros((train_num, 2048))
    var_train_x = np.zeros((train_num, 2048))
    for class_i in np.arange(train_num):
        class_i_x = x[np.argwhere(train_label == train_id[class_i])[:, 0]]
        mu_train_x[class_i] = np.mean(class_i_x, axis=0)
        var_train_x[class_i] = np.var(class_i_x, axis=0)
        class_train_x = class_train_x + [class_i_x]

    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler() #bug huili
    MM_x_S = scaler.fit_transform(x)
    MM_x_test_seen_S = scaler.transform(x_test_seen)
    MM_x_test_S = scaler.transform(x_test)
    max_MM_x = np.max(MM_x_S)

    MM_x = MM_x_S / max_MM_x
    MM_x_test_seen = MM_x_test_seen_S / max_MM_x
    MM_x_test = MM_x_test_S / max_MM_x

    # train set
    train_features = torch.from_numpy(x)
    # train_features=torch.from_numpy(MM_x)
    print(train_features.shape)

    train_label = torch.from_numpy(train_label).unsqueeze(1)
    print(train_label.shape)

    # attributes
    all_attributes = np.array(attribute)
    print(all_attributes.shape)

    attributes = torch.from_numpy(attribute)
    # test set

    test_features = torch.from_numpy(x_test)
    # test_features=torch.from_numpy(MM_x_test)
    print(test_features.shape)

    test_label = torch.from_numpy(test_label).unsqueeze(1)
    print(test_label.shape)

    testclasses_id = np.array(test_id)
    print(testclasses_id.shape)

    test_attributes = torch.from_numpy(att_pro).float()
    print(test_attributes.shape)

    test_seen_features = torch.from_numpy(x_test_seen)
    # test_seen_features = torch.from_numpy(MM_x_test_seen)
    print(test_seen_features.shape)

    test_seen_label = torch.from_numpy(test_label_seen).unsqueeze(1)

    train_data = TensorDataset(train_features, train_label)
    LASSO = models.Ridge(alpha=1)
    LASSO.fit(att_pro.transpose(), att_train.transpose())
    similar = LASSO.coef_
    similar[similar < 1e-3] = 0
    tmp = np.sum(similar, axis=1)
    tmp1 = np.tile(tmp, (similar.shape[1], 1)).transpose()
    similar = similar / tmp1
    sim = torch.from_numpy(similar).float()
    print("training...")
    run = 20
    last_H_zsl = torch.zeros(run)
    last_H_seen = torch.zeros(run)
    last_H_unseen = torch.zeros(run)
    last_H = torch.zeros(run)


    def compute_accuracy(test_features, test_label, test_id_num, test_attributes):

        test_data = TensorDataset(test_features, test_label)
        test_batch = 32
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        label_count = np.zeros(test_id_num.shape[0])
        correct_count = np.zeros(test_id_num.shape[0])

        sample_labels = test_id_num
        sample_attributes = test_attributes
        class_num = sample_attributes.shape[0]
        test_size = test_features.shape[0]

        print("class num:", class_num)

        for batch_features, batch_labels in test_loader:

            batch_size = batch_labels.shape[0]
            batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
            sample_features = attribute_network(Variable(sample_attributes).cuda(GPU).float())
            sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
            relations = relation_network(relation_pairs).view(-1, class_num)

            _, predict_labels = torch.max(relations.data, 1)
            predict_labels = torch.from_numpy(test_id_num[predict_labels])
            batch_labels = batch_labels.squeeze(1)
            count = 0
            for label_id in test_id_num:
                label_id_t = torch.from_numpy(np.array(label_id))
                find_label = (batch_labels == label_id_t)
                find_label_num = torch.sum(find_label)
                if find_label_num.item() > 0:
                    label_count[count] += find_label_num
                    idx = np.nonzero(find_label).squeeze(1)
                    correct_count[count] += accuracy_score(batch_labels[idx], predict_labels[idx],normalize=False)
                count += 1
        acc_use_id = np.nonzero(label_count)
        acc = correct_count[acc_use_id] / label_count[acc_use_id]
        test_accuracy = np.mean(acc)

        return test_accuracy

    def compute_accuracy_grad(test_features, test_label, test_id_num, test_attributes):

        test_data = TensorDataset(test_features, test_label)
        test_batch = 32
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        label_count = np.zeros(test_id_num.shape[0])
        correct_count = np.zeros(test_id_num.shape[0])


        sample_labels = test_id_num
        sample_attributes = test_attributes
        class_num = sample_attributes.shape[0]
        test_size = test_features.shape[0]

        print("class num:", class_num)

        for batch_features, batch_labels in test_loader:

            batch_size = batch_labels.shape[0]

            v_batch_features = Variable(batch_features, requires_grad=True).cuda(GPU).float()  # 32*1024
            v_sample_attributes = Variable(sample_attributes).cuda(GPU).float()
            sample_features = attribute_network(v_sample_attributes)  # k*312

            sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1)
            v_batch_features_ext = v_batch_features.unsqueeze(0).repeat(class_num, 1, 1)
            batch_features_ext = torch.transpose(v_batch_features_ext, 0, 1)

            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
            relations = relation_network(relation_pairs).view(-1, class_num)

            one_hot_labels = Variable(torch.zeros(batch_size, class_num)).cuda(GPU)
            one_hot_labels[:, test_id] = 1

            mse = nn.MSELoss().cuda(GPU)
            loss = mse(relations, one_hot_labels)

            gradient_input_x = \
                torch.autograd.grad(outputs=loss, inputs=v_batch_features, retain_graph=True, only_inputs=True)[0]
            focus_batch_features = - g_rate * gradient_input_x + v_batch_features

            focus_batch_features_ext = focus_batch_features.unsqueeze(0).repeat(class_num, 1, 1)
            focus_batch_features_ext = torch.transpose(focus_batch_features_ext, 0, 1)

            relation_pairs = torch.cat((sample_features_ext, focus_batch_features_ext), 2)
            relations = relation_network(relation_pairs).view(-1, class_num)

            _, predict_labels = torch.max(relations.data, 1)
            predict_labels = torch.from_numpy(test_id_num[predict_labels])
            batch_labels = batch_labels.squeeze(1)
            count = 0
            for label_id in test_id_num:
                label_id_t = torch.from_numpy(np.array(label_id))
                find_label = (batch_labels == label_id_t)
                find_label_num = torch.sum(find_label)
                if find_label_num.item() > 0:
                    label_count[count] += find_label_num
                    idx = np.nonzero(find_label).squeeze(1)
                    correct_count[count] += accuracy_score(batch_labels[idx], predict_labels[idx],
                                                           normalize=False)
                count += 1
        acc_use_id = np.nonzero(label_count)
        acc = correct_count[acc_use_id] / label_count[acc_use_id]
        test_accuracy = np.mean(acc)

        return test_accuracy

    zsl_accuracy = compute_accuracy(test_features, test_label, test_id, test_attributes)
    gzsl_unseen_accuracy = compute_accuracy_grad(test_features, test_label, np.arange(total_class), attributes)
    gzsl_seen_accuracy = compute_accuracy_grad(test_seen_features, test_seen_label, np.arange(total_class), attributes)
    H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

    print('zsl:', zsl_accuracy)
    print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))

if __name__ == '__main__':
    main()