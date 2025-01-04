import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision.models import ResNet18_Weights  # 导入权重枚举类型
import numpy as np  # 导入 numpy

def worker_init_fn(worker_id):
    # 设置随机种子，以避免不同进程之间的随机种子冲突
    import numpy as np
    np.random.seed(worker_id)

def main():
    # 设置数据路径
    train = 'D:/archive/chest_xray/train'
    val = 'D:/archive/chest_xray/val'
    test = 'D:/archive/chest_xray/test'

    # 使用GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据加载
    train_data = datasets.ImageFolder(root=train, transform=train_transform)
    val_data = datasets.ImageFolder(root=val, transform=val_transform)
    test_data = datasets.ImageFolder(root=test, transform=val_transform)

    # 将 num_workers 设置为 0，避免多进程引起的问题
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    # 显示第一批数据
    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(figsize=(10, 10), ncols=4)
    for i in range(4):
        ax = axes[i]
        # 反标准化
        img = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为HWC格式
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反标准化
        img = np.clip(img, 0, 1)  # 将值限制在[0, 1]范围内
        ax.imshow(img)
        ax.set_title(labels[i].item())
    plt.show()

    # 使用预训练的 ResNet18 模型，并更新权重参数
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)  # 修改为使用 'weights' 参数

    # 修改最后一层为二分类
    model.fc = nn.Linear(model.fc.in_features, 1)

    # 将模型迁移到设备
    model = model.to(device)

    # 损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # 训练过程
    epochs = 10
    train_losses = []
    val_losses = []  # 初始化验证损失列表
    test_losses = []  # 初始化测试损失列表

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss_value = criterion(preds, labels.float().unsqueeze(1))
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()

        train_losses.append(train_loss / len(train_loader))

        # 计算验证损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_value = criterion(outputs, labels.float().unsqueeze(1))
                val_loss += loss_value.item()

        val_losses.append(val_loss / len(val_loader))  # 将验证损失添加到列表中

        # 计算测试损失
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_value = criterion(outputs, labels.float().unsqueeze(1))
                test_loss += loss_value.item()

        test_losses.append(test_loss / len(test_loader))  # 将测试损失添加到列表中

        # 测试模型
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 获取模型输出
                predicted = torch.round(torch.sigmoid(outputs))  # 将logits转为0或1
                true_labels.append(labels.cpu().numpy())
                predicted_labels.append(predicted.cpu().numpy())

        true_labels = np.concatenate(true_labels)
        predicted_labels = np.concatenate(predicted_labels)

        # 计算准确率
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # 清理缓存以避免内存泄漏
        torch.cuda.empty_cache()

    # 绘制训练、验证和测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 测试模型
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 获取模型输出
            predicted = torch.round(torch.sigmoid(outputs))  # 将logits转为0或1
            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(predicted.cpu().numpy())

    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)

    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# 使用 if __name__ == '__main__' 来保护主函数
if __name__ == '__main__':
    main()
