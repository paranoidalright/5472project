import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def create_logger(log_dir="logs", log_name="training_log.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    f = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    return log, f

def get_cifar100_loaders(batch_size=128, num_workers=4):
    cifar100_mean = (0.5070751592371323,
                     0.48654887331495095,
                     0.4409178433670343)
    cifar100_std = (0.2673342858792401,
                    0.2564384629170883,
                    0.27615047132568404)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='/home/5472/project/data',
        train=True,
        download=True,
        transform=transform_train
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    testset = torchvision.datasets.CIFAR100(
        root='/home/5472/project/data',
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainloader, testloader

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class HighwayBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.H = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes, 1),
            nn.BatchNorm2d(planes)
        )

        # transform gate T(x)
        self.T = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1,
                      stride=stride, bias=True),
            nn.Sigmoid()
        )

        self.P = None
        if stride != 1 or in_planes != planes:
            self.P = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=stride, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        Hx = self.H(x)
        Tx = self.T(x)

        if self.P is not None:
            Px = self.P(x)
        else:
            Px = x

        out = Hx * Tx + Px * (1.0 - Tx)
        out = self.relu(out)
        return out


class InceptionResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        base = max(planes // 6, 1)
        c1 = base
        c2 = 2 * base
        c3 = 2 * base
        c4 = planes - (c1 + c2 + c3)
        if c4 <= 0:
            c1 = planes // 4
            c2 = planes // 4
            c3 = planes // 4
            c4 = planes - (c1 + c2 + c3)

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes, c1, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, c2, kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, c3, kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 avg pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_planes, c4, kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )

        self.out_bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.out_bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CifarResidualNet(nn.Module):
    def __init__(self, block, depth=20, num_classes=100, widen_factor=1):
        super().__init__()
        assert (depth - 2) % 6 == 0,
        n = (depth - 2) // 6

        base_channels = 16
        self.in_planes = base_channels * widen_factor

        self.conv1 = conv3x3(3, self.in_planes)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block,
                                       base_channels * widen_factor,
                                       n, stride=1)
        self.layer2 = self._make_layer(block,
                                       base_channels * 2 * widen_factor,
                                       n, stride=2)
        self.layer3 = self._make_layer(block,
                                       base_channels * 4 * widen_factor,
                                       n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4 * widen_factor, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def build_model(arch: str, depth: int, num_classes: int = 100, widen_factor: int = 1):
    if arch == 'plain':
        return CifarResidualNet(PlainBlock,
                                depth=depth,
                                num_classes=num_classes,
                                widen_factor=1)
    elif arch == 'resnet':
        return CifarResidualNet(BasicBlock,
                                depth=depth,
                                num_classes=num_classes,
                                widen_factor=1)
    elif arch == 'highway':
        return CifarResidualNet(HighwayBlock,
                                depth=depth,
                                num_classes=num_classes,
                                widen_factor=1)
    elif arch == 'inception':
        # Inception-Res block
        return CifarResidualNet(InceptionResBlock,
                                depth=depth,
                                num_classes=num_classes,
                                widen_factor=1)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def get_grad_layers(model: nn.Module):
    layers = []
    for name, p in model.named_parameters():
        if p.requires_grad and 'weight' in name:
            layers.append(p)
    return layers


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_sum += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


def train_one_model(model, trainloader, testloader, device, num_epochs=200, lr=0.1, weight_decay=5e-4, milestones=(100, 150), log=None, desc=""):
    if log is None:
        def log(x): print(x)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=0.1
    )

    grad_layers = get_grad_layers(model)

    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []
    grad_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        grad_sq_sums = [0.0 for _ in grad_layers]
        grad_counts = [0 for _ in grad_layers]

        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for idx, p in enumerate(grad_layers):
                if p.grad is not None:
                    g = p.grad.detach().view(-1)
                    grad_sq_sums[idx] += torch.dot(g, g).item()
                    grad_counts[idx] += 1

            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            running_total += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        test_loss, test_acc = evaluate(model, testloader, device)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        epoch_grad_norms = []
        for s, c in zip(grad_sq_sums, grad_counts):
            if c > 0:
                epoch_grad_norms.append(math.sqrt(s / c))
            else:
                epoch_grad_norms.append(0.0)
        grad_history.append(epoch_grad_norms)

        log(f"{desc} Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    best_acc = max(test_acc_history)
    best_acc_epoch = test_acc_history.index(best_acc) + 1
    best_loss = min(test_loss_history)
    best_loss_epoch = test_loss_history.index(best_loss) + 1
    log(f"{desc} Best Test Acc: {best_acc:.4f} at epoch {best_acc_epoch}")
    log(f"{desc} Min Test Loss: {best_loss:.4f} at epoch {best_loss_epoch}")

    return {
        'train_acc': train_acc_history,
        'test_acc': test_acc_history,
        'train_loss': train_loss_history,
        'test_loss': test_loss_history,
        'grad_history': grad_history,
    }



def get_depth_color_map(depths):
    depths_sorted = sorted(depths)
    colors = plt.cm.tab5.colors 
    depth_color = {}
    for i, d in enumerate(depths_sorted):
        depth_color[d] = colors[i % len(colors)]
    return depth_color


def plot_experiment1(results, depths_exp1, out_dir, log=None):
    if log is None:
        def log(x): print(x)

    os.makedirs(out_dir, exist_ok=True)
    depth_color = get_depth_color_map(depths_exp1)

    # fig1
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    arch_order = ['plain', 'resnet']
    titles1 = {
        'plain': 'PlainNet (20/56/80)',
        'resnet': 'ResNet (20/56/80)'
    }

    for ax, arch in zip(axes1.flat, arch_order):
        for depth in depths_exp1:
            hist = results[(arch, depth)]
            epochs = range(1, len(hist['train_loss']) + 1)
            train_err = [1.0 - acc for acc in hist['train_acc']]
            test_err = [1.0 - acc for acc in hist['test_acc']]
            c = depth_color[depth]

            ax.plot(epochs, train_err,
                    linestyle='--', color=c,
                    label=f"Train (depth={depth})")
            ax.plot(epochs, test_err,
                    linestyle='-', color=c,
                    label=f"Test (depth={depth})")

        ax.set_title(titles1[arch])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (1 - accuracy)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "exp1_error.png")
    fig1.savefig(fig1_path, dpi=200)
    log(f"[Experiment 1] Saved train/test error figure to {fig1_path}")

    # fig2
    n_depths = len(depths_exp1)
    fig2, axes2 = plt.subplots(n_depths, 2,
                               figsize=(12, 4 * n_depths),
                               squeeze=False)
    titles2 = {
        'plain': 'PlainNet',
        'resnet': 'ResNet'
    }

    for i, depth in enumerate(sorted(depths_exp1)):
        for j, arch in enumerate(arch_order):
            ax = axes2[i][j]
            hist = results[(arch, depth)]
            grad_hist = hist['grad_history']
            grad_mat = torch.tensor(grad_hist)

            im = ax.imshow(grad_mat.cpu().numpy(),
                           aspect='auto',
                           origin='lower')
            ax.set_title(f"{titles2[arch]} - depth {depth}")
            ax.set_xlabel("Layer index (weight params order)")
            ax.set_ylabel("Epoch")
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2_path = os.path.join(
        out_dir,
        "exp1_grad_heatmap.png"
    )
    fig2.savefig(fig2_path, dpi=200)


def plot_experiment2(results, depths_exp2, out_dir, log=None):
    if log is None:
        def log(x): print(x)

    os.makedirs(out_dir, exist_ok=True)
    depth_color = get_depth_color_map(depths_exp2)

    arch_order = ['resnet', 'highway', 'inception']
    titles = {
        'resnet': "ResNet",
        'highway': "HighwayNet",
        'inception': "Inception-Res"
    }

    # fig1
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    for ax, arch in zip(axes1.flat, arch_order):
        for depth in depths_exp2:
            hist = results[(arch, depth)]
            epochs = range(1, len(hist['train_loss']) + 1)
            # train_loss = hist['train_loss']
            # test_loss = hist['test_loss']
            train_err = [1.0 - acc for acc in hist['train_acc']]
            test_err = [1.0 - acc for acc in hist['test_acc']]
            c = depth_color[depth]

            ax.plot(epochs, train_err,
                    linestyle='--', color=c,
                    label=f"Train (depth={depth})")
            ax.plot(epochs, test_err,
                    linestyle='-', color=c,
                    label=f"Test (depth={depth})")

        ax.set_title(f"{titles[arch]} (depth 20/56/80)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

    fig1.tight_layout()
    fig1_path = os.path.join(
        out_dir,
        "exp2_error.png"
    )
    fig1.savefig(fig1_path, dpi=200)
    log(f"[Experiment 2] Saved train/test loss figure to {fig1_path}")

    # fig2
    n_depths = len(depths_exp2)
    fig2, axes2 = plt.subplots(n_depths, 3,
                               figsize=(18, 4 * n_depths),
                               squeeze=False)

    for i, depth in enumerate(sorted(depths_exp2)):
        for j, arch in enumerate(arch_order):
            ax = axes2[i][j]
            hist = results[(arch, depth)]
            grad_hist = hist['grad_history']
            grad_mat = torch.tensor(grad_hist)

            im = ax.imshow(grad_mat.cpu().numpy(),
                           aspect='auto',
                           origin='lower')
            ax.set_title(f"{titles[arch]} - depth {depth}")
            ax.set_xlabel("Layer index (weight params order)")
            ax.set_ylabel("Epoch")
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2_path = os.path.join(
        out_dir,
        "exp2_grad_heatmap.png"
    )
    fig2.savefig(fig2_path, dpi=200)
    log(f"[Experiment 2] Saved grad heatmap (all depths) to {fig2_path}")


def main():
    log, log_file = create_logger(
        log_dir="/home/5472/project/logs_4",
        log_name="training_log.txt"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    batch_size = 128
    num_epochs = 200
    trainloader, testloader = get_cifar100_loaders(batch_size=batch_size)
    results = {}

    # exp1
    depths_exp1 = [20, 56, 80]

    for arch in ['plain', 'resnet']:
        for depth in depths_exp1:
            key = (arch, depth)
            if key in results:
                continue
            log("=" * 80)
            log(f"[Experiment 1] Training arch={arch}, depth={depth}")
            log("=" * 80)
            model = build_model(arch=arch, depth=depth, num_classes=100)
            history = train_one_model(
                model,
                trainloader,
                testloader,
                device,
                num_epochs=num_epochs,
                lr=0.1,
                weight_decay=5e-4,
                milestones=(100, 150),
                log=log,
                desc=f"[Exp1][{arch}-{depth}]"
            )
            results[key] = history

    plot_experiment1(
        results,
        depths_exp1=depths_exp1,
        out_dir="/home/5472/project/figures_exp1_4",
        log=log
    )

    # exp2
    depths_exp2 = [20, 56, 80]
    exp2_arches = ['resnet', 'highway', 'inception']

    for arch in exp2_arches:
        for depth in depths_exp2:
            key = (arch, depth)
            if key in results:
                continue
            log("=" * 80)
            log(f"[Experiment 2] Training arch={arch}, depth={depth}")
            log("=" * 80)
            model = build_model(
                arch=arch,
                depth=depth,
                num_classes=100
            )
            history = train_one_model(
                model,
                trainloader,
                testloader,
                device,
                num_epochs=num_epochs,
                lr=0.1,
                weight_decay=5e-4,
                milestones=(100, 150),
                log=log,
                desc=f"[Exp2][{arch}-{depth}]"
            )
            results[key] = history

    plot_experiment2(
        results,
        depths_exp2=depths_exp2,
        out_dir="/home/5472/project/figures_exp2",
        log=log
    )
    log_file.close()

if __name__ == "__main__":
    main()
