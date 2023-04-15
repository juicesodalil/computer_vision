import time
import torch


def train(epochs, model, train_loader, criterion, optimizer, scheduler, device, network_name):
    start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            # 修改学习率

            train_loss += loss.item()
            _, prediction = out.max(1)
            total += targets.size(0)
            correct += prediction.eq(targets).sum().item()

            print(batch_idx + 1, '/', len(train_loader), 'epoch: %d' % epoch,
                  '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        scheduler.step()
        # if epoch != 1 and epoch % 10 == 0:
        print('\t last_lr:', scheduler.get_last_lr())

    finish = time.time()
    print('total time : %.2f' % (finish - start))
    torch.save(model.state_dict(), f"./pretrain/cifar10-{network_name}-{epochs}.pth")


def test(model, test_loader, criterion, device, network_name):
    start = time.time()
    model.eval()
    sum = 0.0
    acc = 0.0
    loss_sum = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        loss = criterion(out, targets)
        pred = out.argmax(dim=1, keepdims=True)
        acc += pred.eq(targets.view_as(pred)).sum().item()
        sum += len(targets)
        loss_sum += loss.item()

        print(batch_idx + 1, '/', len(test_loader),
              '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (loss_sum / (batch_idx + 1), 100. * acc / sum, acc, sum))

    finish = time.time()
    print('total time : %.2f' % (finish - start))
