import os
import math
import random
from collections import defaultdict
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


def get_cifar100_loaders(batch_size=128, num_workers=4, data_root="./data"):
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
        root=data_root,
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
        root=data_root,
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


def count_total_blocks(model):
    total = 0
    for layer in [model.layer1, model.layer2, model.layer3]:
        total += len(layer)
    return total


def get_block_index_info(model):
    info = []
    idx = 0
    for layer_name in ["layer1", "layer2", "layer3"]:
        layer = getattr(model, layer_name)
        for i, b in enumerate(layer):
            info.append((idx, layer_name, i, b))
            idx += 1
    return info


def get_safe_block_indices(model):
    safe = []
    for idx, layer_name, i, b in get_block_index_info(model):
        if isinstance(b, BasicBlock):
            # conv1 stride
            stride = b.conv1.stride[0] if isinstance(b.conv1, nn.Conv2d) else 1
            if b.downsample is None and stride == 1:
                safe.append(idx)
        else:
            pass
    return safe


def forward_with_drops(model, x, drop_block_indices):
    out = model.conv1(x)
    out = model.bn1(out)
    out = model.relu(out)

    idx = 0 

    # layer1
    for b in model.layer1:
        if idx in drop_block_indices:
            out = out
        else:
            out = b(out)
        idx += 1

    # layer2
    for b in model.layer2:
        if idx in drop_block_indices:
            out = out
        else:
            out = b(out)
        idx += 1

    # layer3
    for b in model.layer3:
        if idx in drop_block_indices:
            out = out
        else:
            out = b(out)
        idx += 1

    out = model.avgpool(out)
    out = torch.flatten(out, 1)
    out = model.fc(out)
    return out


def evaluate_with_drops(model, testloader, device, drop_block_indices):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = forward_with_drops(model, inputs, drop_block_indices)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = correct / total
    return acc


def main():
    DATA_ROOT = "/home/5472/project/data"
    LOG_DIR = "/home/5472/project/logs_resnet200_ensemble_1"
    CKPT_DIR = "/home/5472/project/checkpoints_resnet200_ensemble_1"
    FIG_DIR = "/home/5472/project/figures_resnet200_ensemble_1"

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    CKPT_PATH = os.path.join(CKPT_DIR, "resnet200_cifar100.pth")

    BATCH_SIZE = 128
    NUM_WORKERS = 4
    DEPTH = 80
    NUM_CLASSES = 100
    NUM_EPOCHS = 200
    MAX_DELETE = 10 
    NUM_TRIALS_PER_K = 30 
    RANDOM_SEED = 2025

    log, log_file = create_logger(LOG_DIR, "log_resnet200_ensemble.txt")

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    trainloader, testloader = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_root=DATA_ROOT
    )

    model = CifarResidualNet(BasicBlock,
                             depth=DEPTH,
                             num_classes=NUM_CLASSES,
                             widen_factor=1)

    if os.path.isfile(CKPT_PATH):
        log(f"Checkpoint found, loading from {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        log("=" * 80)
        log(f"Start training ResNet-{DEPTH} on CIFAR-100 from scratch")
        log("=" * 80)

        history = train_one_model(
            model,
            trainloader,
            testloader,
            device,
            num_epochs=NUM_EPOCHS,
            lr=0.1,
            weight_decay=5e-4,
            milestones=(100, 150),
            log=log,
            desc=f"[ResNet{DEPTH}]"
        )

        save_obj = {
            "state_dict": model.state_dict(),
            "history": history,
        }
        torch.save(save_obj, CKPT_PATH)
        log(f"Checkpoint saved to {CKPT_PATH}")

    model = model.to(device)

    total_blocks = count_total_blocks(model)
    safe_indices = get_safe_block_indices(model)

    base_loss, base_acc = evaluate(model, testloader, device)
    base_error = 1.0 - base_acc
    log(f"Baseline (no deletion) - "
        f"Loss: {base_loss:.4f}, Acc: {base_acc:.4f}, Error: {base_error:.4f}")

    errors_by_k = defaultdict(list)
    for k in range(1, MAX_DELETE + 1):
        actual_k = min(k, len(safe_indices))
        log(f"\n=== K = {k} deleted blocks (actual_k={actual_k}) ===")
        for t in range(NUM_TRIALS_PER_K):
            drop_indices = set(random.sample(safe_indices, actual_k))
            acc = evaluate_with_drops(model, testloader, device, drop_indices)
            error = 1.0 - acc
            errors_by_k[k].append(error)
            log(f"  [K={k}] Trial {t+1:02d}/{NUM_TRIALS_PER_K}: "
                f"Acc = {acc:.4f}, Error = {error:.4f}")

    ks = sorted(errors_by_k.keys())
    data = [errors_by_k[k] for k in ks]

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(
        data,
        positions=ks,
        widths=0.6,
        patch_artist=True,
        showfliers=False
    )

    for patch in box['boxes']:
        patch.set_alpha(0.6)

    plt.axhline(base_error, linestyle="--", color="gray",
                label=f"Baseline error (no deletion): {base_error:.3f}")

    plt.xlabel("Number of deleted residual blocks (K)")
    plt.ylabel("Test error (1 - accuracy)")
    plt.title(f"ResNet-{DEPTH} on CIFAR-100: Randomly Deleting Residual Blocks at Test Time")
    plt.xticks(ks)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    fig_path = os.path.join(
        FIG_DIR,
        f"resnet{DEPTH}_boxplot_{k}.png"
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    log(f"\nSaved boxplot to {fig_path}")
    log_file.close()

if __name__ == "__main__":
    main()
