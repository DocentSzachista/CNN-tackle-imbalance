import os
# from utils import progress_bar

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from utils import IMAGE_PREPROCESSING
from models.resnet import ResNet101, ResNet
from dataset_downloader import reduce_dataset_over_strategy, load_subset_dataset
from metrics import generate_confussion_matrix
from imbalance_strategies import *
from dataset_downloader import SCENARIO_1, SCENARIO_2, SCENARIO_3


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=IMAGE_PREPROCESSING)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=IMAGE_PREPROCESSING)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


net = ResNet101()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(f"{batch_idx} {len(trainloader)} : Loss {round((train_loss / (batch_idx + 1)), 2)} | Acc: {round(100. * correct / total, 2)}  ({correct}, {total})")


def test(epoch, model_save_path, debug: bool | None = False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    current_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_loss = round((test_loss / (batch_idx + 1)), 2)
            if debug:
                print(
                    f" Test {batch_idx} {len(trainloader)} : Loss {current_loss} | Acc: {round(100. * correct / total, 2)}  ({correct}, {total})")

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        save_model(net, model_save_path, acc, epoch, current_loss)
        # if not os.path.isdir('checkpoint'):
        # os.mkdir('checkpoint')
        best_acc = acc


def save_model(net: ResNet, save_path: str, acc: float, epoch: int, current_loss: float):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'loss': current_loss
    }
    torch.save(state, save_path)


def load_model(checkpoint_path: str):
    global device

    model = ResNet101()
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    model.load_state_dict(
        checkpoint['net']
    )
    return model


def run(model_path: str, data_path: str = "", imbalance_reduction_strategies: dict):
    global trainset
    global trainloader
    global criterion


    if IN_COLAB:
        model_path = "/content/drive/MyDrive/" + model_path

    if data_path:
        load_subset_dataset(trainloader, data_path)
        print("dane inne")

    if imbalance_reduction_strategies.get("weights", False):
        classes_ratio = imbalance_reduction_strategies['weight_values']
        single_class_ammount = imbalance_reduction_strategies['amount']
        classes_count = [
             int(x * single_class_ammount) for x in classes_ratio
        ]
        classes_sum = sum(classes_count)
        classes_weights = [ classes_sum / x for x in classes_count]
        get_criterion_loss( classes_weights )
    if imbalance_reduction_strategies.get("show_often", False):
        retrieve_weighed_dataloader(trainloader)


    for epoch in range(start_epoch, start_epoch + 55):
        train(epoch)
        test(epoch, model_path)
        scheduler.step()


def test_model(model_path: str, output_name: str):
    global trainset
    global trainloader
    global device

    model = load_model(model_path)
    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    total = 0
    current_loss = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_loss = round((test_loss / (batch_idx + 1)), 2)

            # add batch labels to lists
            y_pred.extend(
                predicted.data.cpu().numpy()
            )
            y_true.extend(
                targets.data.cpu().numpy()
            )
            print(f" Test {batch_idx} {len(trainloader)} : Loss {current_loss} | Acc: {round(100. * correct / total, 2)}  ({correct}, {total})")
        generate_confussion_matrix(y_true, y_pred, output_name)


if __name__ == "__main__":
    # Options related to Windows, uncomment if you use that env
    # torch.backends.cudnn.benchmark = True  # You can enable this for better performance
    # torch.cuda.set_per_process_memory_fraction(0.7)  # Adjust the fraction as needed
    # # Set max_split_size_mb (e.g., to 512MB)
    # torch.set_default_tensor_type(torch.FloatTensor)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.empty_cache()

    # Uncomment if you want to train model
    # run("./strategy_two_class.ckpt")

    # Uncomment if you want to retrieve metrics from trained model
    test_model(
        "./", "test"
    )
