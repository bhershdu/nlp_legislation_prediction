import torch
import pandas as pd
import os
import fnmatch
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class TitlePartyModel(torch.nn.Module):
    """
    Simple NN using Sigmoid activations
    """
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(2048,2048, dtype=torch.float32)
        self.input_activation = torch.nn.Sigmoid()
        self.hidden1 = torch.nn.Linear(2048,1024)
        self.hidden1_activation = torch.nn.Sigmoid()
        self.hidden2 = torch.nn.Linear(1024,128)
        self.hidden2_activation = torch.nn.Sigmoid()
        # 4 political party choices
        self.hidden3 = torch.nn.Linear(128,4)
        self.output = torch.nn.Softmax()

    def forward(self, x):
        x = self.input(x)
        x = self.input_activation(x)
        x = self.hidden1(x)
        x = self.hidden1_activation(x)
        x = self.hidden2(x)
        x = self.hidden2_activation(x)
        x = self.hidden3(x)
        x = self.output(x)
        return x

class TitlePartyModelRelu(torch.nn.Module):
    """
    Simple model using ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(2048,2048, dtype=torch.float32)
        self.input_activation = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(2048,1024)
        self.hidden1_activation = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(1024,128)
        self.hidden2_activation = torch.nn.ReLU()
        # 4 political party choices
        self.hidden3 = torch.nn.Linear(128,4)
        self.output = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = self.input_activation(x)
        x = self.hidden1(x)
        x = self.hidden1_activation(x)
        x = self.hidden2(x)
        x = self.hidden2_activation(x)
        x = self.hidden3(x)
        x = self.output(x)
        return x

class TitlePartyModelReluCollapse(torch.nn.Module):
    """
    Similar to TitlePartyModelRelu but allow the number of outputs to be specified.
    This was an experiment in collapsing the labels, primarily removing missing party value.
    """
    def __init__(self, num_labels):
        super().__init__()
        self.input = torch.nn.Linear(2048,2048, dtype=torch.float32)
        self.input_activation = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(2048,1024)
        self.hidden1_activation = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(1024,128)
        self.hidden2_activation = torch.nn.ReLU()
        # 4 political party choices
        self.hidden3 = torch.nn.Linear(128,num_labels)
        self.output = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = self.input_activation(x)
        x = self.hidden1(x)
        x = self.hidden1_activation(x)
        x = self.hidden2(x)
        x = self.hidden2_activation(x)
        x = self.hidden3(x)
        x = self.output(x)
        return x

class SimilarityFilteredSummaryDataSet(torch.utils.data.Dataset):
    """
    Pytorch dataset that attempts to only include files above some cosine-similarity threshold.
    This experiment didn't yield anything. Probably because I should have done a less-then check instead of a greater-than
    check.
    """
    def __init__(self, file_path_arr, fixed_idx=False, allow_gpu=False, col_name="input_maxpool", similarity_threshold=0.5):
        """
        Init the class
        :param file_path_arr:  a list with the file paths to consider
        :param fixed_idx: if True, only return the 0th indexed data
        :param allow_gpu: if True, use the gpu for the Tensors
        :param col_name: the column name to find the data in the data frame
        :param similarity_threshold: the cosine-similarity threshold
        """
        self.data_frames = []
        self.fixed_idx = fixed_idx
        self.allow_gpu = allow_gpu
        self.col_name = col_name
        self.reference_array = None
        for f in file_path_arr:
            print(f"loading {f}")
            data_df = pd.read_pickle(f, compression="gzip")
            if self.reference_array is None:
                self.reference_array = np.array(data_df[col_name])
                self.data_frames.append(data_df)
            else:
                current_array = data_df[col_name]
                dist = cosine_similarity([self.reference_array], [np.array(current_array)])
                print(dist)
                if max(dist) > similarity_threshold:
                    print(f"adding {f}")
                    self.data_frames.append(data_df)
                else:
                    print(f'rejecting {f}')
        if torch.cuda.is_available() and allow_gpu:
            self.t_device = f'cuda:{torch.cuda.current_device()}'
        else:
            self.t_device = 'cpu'
        print(f"device set to {self.t_device}")

    def __len__(self):
        if self.fixed_idx:
            return 1
        else:
            return len(self.data_frames)

    def __getitem__(self, idx):
        if self.fixed_idx:
            next_df = self.data_frames[0]
        else:
            next_df = self.data_frames[idx]
        party = next_df["party"][0]  # they are all the same, so just pick the first one
        encoding = torch.tensor(np.array(next_df[self.col_name]), dtype=torch.float, device=self.t_device)
        # 4 politcal party choices
        party_arr = np.zeros(4, dtype=int)
        # the party index was stored as value with a starting index of 1 -- rethink this
        party_arr[party - 1] = 1  # set the value to 1 for the party index
        return encoding, torch.tensor(party_arr, dtype=torch.float, device=self.t_device)

class SimilarityFilteredSummaryDataSetByLabel(torch.utils.data.Dataset):
    """
    Calculate cosine-similarity against a reference data point per label.
    """
    def __init__(self, file_path_arr, fixed_idx=False, allow_gpu=False, col_name="input_maxpool", similarity_threshold=0.5):
        """
        Init the class
        :param file_path_arr:  a list with the file paths to consider
        :param fixed_idx: if True, only return the 0th indexed data
        :param allow_gpu: if True, use the gpu for the Tensors
        :param col_name: the column name to find the data in the data frame
        :param similarity_threshold: the cosine-similarity threshold
        """
        self.data_frames = []
        self.fixed_idx = fixed_idx
        self.allow_gpu = allow_gpu
        self.col_name = col_name
        self.label_reference_array = {}
#        self.reference_array = None
        if torch.cuda.is_available() and allow_gpu:
            self.t_device = f'cuda:{torch.cuda.current_device()}'
        else:
            self.t_device = 'cpu'
        print(f"device set to {self.t_device}")
        for f in file_path_arr:
            print(f"loading {f}")
            data_df = pd.read_pickle(f, compression="gzip")
            l_array = np.array(data_df["party"])
            # print(l_array)
            i_index = np.argmax(l_array, axis=0)
            i_index_str = str(i_index)
            include = True
            if i_index_str not in self.label_reference_array:
                self.label_reference_array[i_index_str] = np.array(data_df[col_name])
            else:
                current_array = data_df[col_name]
                dist = cosine_similarity([self.label_reference_array[i_index_str]], [np.array(current_array)])
                include = dist > similarity_threshold
            if include:
                if self.data_frames is None:
                    self.data_frames = data_df
                else:
                    self.data_frames.append(data_df)
            else:
                print(f"excluding {f} with label index {i_index}")

    def __len__(self):
        if self.fixed_idx:
            return 1
        else:
            return len(self.data_frames)

    def __getitem__(self, idx):
        if self.fixed_idx:
            next_df = self.data_frames[0]
        else:
            next_df = self.data_frames[idx]
        party = next_df["party"][0]  # they are all the same, so just pick the first one
        encoding = torch.tensor(np.array(next_df[self.col_name]), dtype=torch.float, device=self.t_device)
        # 4 politcal party choices
        party_arr = np.zeros(4, dtype=int)
        # the party index was stored as value with a starting index of 1 -- rethink this
        party_arr[party - 1] = 1  # set the value to 1 for the party index
        return encoding, torch.tensor(party_arr, dtype=torch.float, device=self.t_device)


class SummaryDataSet(torch.utils.data.Dataset):
    """
    Use the dataset to remove one of the party label values.
    """
    def __init__(self, file_path_arr, fixed_idx=False, allow_gpu=False, col_name="input_maxpool", collapse_label=False, remove_index=0):
        """
        Init the class
        :param file_path_arr:  a list with the file paths to consider
        :param fixed_idx: if True, only return the 0th indexed data
        :param allow_gpu: if True, use the gpu for the Tensors
        :param collapse_label: if True, remove the party label index specified next
        :param remove_index: the party label index to remove
        """
        self.data_frames = []
        self.fixed_idx = fixed_idx
        self.allow_gpu = allow_gpu
        self.col_name = col_name
        self.collapse_label = collapse_label
        self.remove_index = remove_index
        for f in file_path_arr:
            print(f"loading {f}")
            self.data_frames.append(pd.read_pickle(f, compression="gzip"))
        if torch.cuda.is_available() and allow_gpu:
            self.t_device = f'cuda:{torch.cuda.current_device()}'
        else:
            self.t_device = 'cpu'
        print(f"device set to {self.t_device}")


    def __len__(self):
        if self.fixed_idx:
            return 1
        else:
            return len(self.data_frames)

    def __getitem__(self, idx):
        if self.fixed_idx:
            next_df = self.data_frames[0]
        else:
            next_df = self.data_frames[idx]
        party = next_df["party"][0] # they are all the same, so just pick the first one
        encoding = torch.tensor(np.array(next_df[self.col_name]),dtype=torch.float, device=self.t_device)
        # 4 politcal party choices
        party_arr = np.zeros(4,dtype=int)
        # the party index was stored as value with a starting index of 1 -- rethink this
        party_arr[party-1] = 1 # set the value to 1 for the party index
        if self.collapse_label:
            party_arr = np.delete(party_arr, self.remove_index)

        return encoding, torch.tensor(party_arr,dtype=torch.float, device=self.t_device)


def train_one_epoch(model, loss_function, the_optimizer, summary_writer, training_dataloader, scheduler):
    """
    Train one epoch of the model
    :param model: the model instance
    :param loss_function: the loss function instance
    :param the_optimizer: the optimizer instance
    :param summary_writer: the summary writer instance
    :param training_dataloader: the training dataloader
    :param scheduler: the optional scheduler (for learning rate reduction)
    :return: the last loss seen
    """
    running_loss = 0
    last_loss = 0.
    for i, data in enumerate(training_dataloader):
#        print(f"data index {i}")
        inputs, label = data
        input_tensor = inputs.view(1,-1)
        the_optimizer.zero_grad()
        outputs = model(input_tensor)
        if i % 100 == 0:
            print(f'{label} vs {outputs}')
        loss = loss_function(outputs, label)
        if i % 100 == 0:
            print(f'last loss = {loss.item()}')
        last_loss = loss.item()
        loss.backward()
        the_optimizer.step()
        summary_idx = i * len(training_dataloader) + i + 1
        summary_writer.add_scalar("loss/train", last_loss, summary_idx)
    return last_loss


def run_n_epochs(max_epoch,
                 model,
                 loss_function,
                 the_optimizer,
                 training_dataloader,
                 validation_dataloader,
                 checkpoint_name,
                 scheduler,
                 return_learning_rates=False):
    """
    Run n epochs
    :param max_epoch: the maximum number of epochs to run
    :param model: the model instance
    :param loss_function: the loss function instance
    :param the_optimizer: the optimizer instance
    :param summary_writer: the summary writer instance
    :param training_dataloader: the training dataloader
    :param validation_dataloader: the vailidation dataloader
    :param checkpoint_name: the name of the model checkpoint file written at the end of n epochs
    :param scheduler: the optional scheduler (for learning rate reduction)
    :param return_learning_rates: if True the return will include the learning rates in the output
    :return: a tuple of the checkpoint name (timestamped), the training losses, the validation losses, and the learning rates
    """
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(start_time))
    epoch_num = 0
    val_losses = []
    train_losses = []
    learning_rates = []
    for epoch in range(max_epoch):
        print(f"epoch {epoch}")
        print("turn on training")
        model.train(True)
        print("running one epoch")
        last_epoch_loss = train_one_epoch(model,loss_function, the_optimizer, writer, training_dataloader, scheduler)
        train_losses.append(last_epoch_loss)
        print("turn off training")
        model.train(False)
        print(f'epoch loss {last_epoch_loss}')

        # title_model.train(False)

        running_validation_loss = 0.0
        print("applying model.eval()")
        model.eval()
        with (torch.no_grad()):
            for i, vdata in enumerate(validation_dataloader):
                # print(f"validation {i}")
                vinputs, vlabel = vdata
                # print(f"label : {vlabel}")
                voutputs = model(vinputs)
                # print(f"vOutput: {voutputs}")
                vloss = loss_function(voutputs, vlabel)
                running_validation_loss += vloss

        avg_vloss = running_validation_loss / len(validation_dataloader)
        val_losses.append(avg_vloss)
        print('LOSS train {} valid {}'.format(last_epoch_loss, avg_vloss))
    #    writer.add_scalars("Training vs Valiation loss",{"training": last_epoch_loss, "validation": avg_vloss}, epoch_num+1)
    #    writer.flush()
        epoch_num += 1
        if scheduler != None:
            print("stepping scheduler")
            learning_rates.append(scheduler.get_last_lr())
            scheduler.step()
    save_name = f'{checkpoint_name}_{start_time}.pkl'
    torch.save(model.state_dict(), save_name)
    if return_learning_rates:
        return [save_name, train_losses, val_losses, learning_rates]
    else:
        return [save_name, train_losses, val_losses]

def split_data(token_path, file_keyword, train_split, party_filter=None):
    """
    Split the data into training, validation, and test
    :param token_path: the directory where the tokenized data is located
    :param file_keyword: filter for file names with the substring
    :param train_split: the training split value. validation and test are split equally from the remainder
    :param party_filter: optionally filter the data to only 1 party value
    :return: tuple of the train files, validation files, and test file paths
    """
    train_files = []
    validate_files = []
    test_files = []
    for root, dirs, files in os.walk(token_path):
        for f in files:
            if fnmatch.fnmatch(f, f'*{file_keyword}*'):
                df = pd.read_pickle(os.path.join(root, f), compression="gzip")
                if party_filter is None or df['party'][0] == party_filter:
                    if np.random.sample(1) <= train_split:
                        print(f'train: {f}')
                        train_files.append(os.path.join(root, f))
                    elif np.random.sample(1) < 0.5:
                        print(f'validate {f}')
                        validate_files.append(os.path.join(root, f))
                    else:
                        print(f'test : {f}')
                        test_files.append(os.path.join(root, f))
    return [train_files, validate_files, test_files]


def split_data_from_array(filepath_arr, train_split):
    """
    Given an array of file paths, split the data
    :param filepath_arr: the array of file paths
    :param train_split: the train split value.
    :return: tuple of the train files, validation files, and test file paths
    """
    train_files = []
    validate_files = []
    test_files = []
    for f in filepath_arr:
        if np.random.sample(1) <= train_split:
            train_files.append(f)
        elif np.random.sample(1) < 0.5:
            validate_files.append(f)
        else:
            test_files.append(f)
    return [train_files, validate_files, test_files]


def plot_losses(train_losses, validation_losses):
    """
    Plot the training and validation losses in a standard way
    :param train_losses: training loss array
    :param validation_losses: validation los array
    :return: None
    """
    if isinstance(validation_losses[0], torch.Tensor):
        validation_loses_float = list(map(lambda x: x.item(), validation_losses))
    else:
        validation_loses_float = validation_losses
    fig, (plt1, plt2) = plt.subplots(1, 2)
    plt1.plot(range(len(train_losses)), train_losses)
    plt1.set_yscale('log')
    plt1.set_title("training loss vs epoch")
    plt2.plot(range(len(validation_loses_float)), validation_loses_float)
    plt2.set_yscale('log')
    plt2.set_title("validation loss vs epoch")

def get_test_results(model, dataloader):
    """
    Get the test data inference results
    :param model: the trained model
    :param dataloader: the test dataloader
    :return: dataframe with the inference results
    """
    element_data = None
    with (torch.no_grad()):
        for i, tdata in enumerate(dataloader):
            # print(f"validation {i}")
            tinputs, tlabel = tdata
            # print(tlabel[0])
            l_array = np.array(tlabel[0].cpu())
            # print(l_array)
            i_index = np.argmax(l_array, axis=0)
            # print(i_index)
            # print(f'expected label index : {i_index} from {l_array}')
            # print(f"label : {vlabel}")
            toutputs = model(tinputs)
            o_array = np.array(toutputs[0].cpu())
            o_index = np.argmax(o_array, axis=0)
            # print(f'inferred label index : {o_index} from {o_array}')
            element = {"input": [l_array],
                       "output": [o_array],
                       "expected_index": [i_index],
                       "inferred_index": [o_index]}
            if element_data is None:
                element_data = element
            else:
                element_data["input"].append(l_array)
                element_data["output"].append(o_array)
                element_data["expected_index"].append(i_index)
                element_data["inferred_index"].append(o_index)
                print(len(element_data))
    return pd.DataFrame(element_data)

