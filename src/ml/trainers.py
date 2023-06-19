import math
import torch
from torch import nn
import numpy as np
from tqdm.notebook import tqdm

def train_on_batch(model, batch_ids, batch_masks, batch_y, optimizer, scheduler, loss_function):
    """
    TODO
    """
    # переводим модель в режим обучения
    model.train()
    # обнуляем градиенты (тк они ненулевые из-за ранних обучений на других батчах)
    model.zero_grad()
    # передаем батч -> обучаем BackProp-ом -> выводим предсказание
    pred = model(batch_ids, batch_masks)
    # считаем ошибку через loss_function
    loss = loss_function(pred, batch_y)
    loss.backward()
    # делаем шаг оптимизатора и scheduler
    optimizer.step()
    if scheduler != None:
        scheduler.step()
    # возвращаем ошибку
    return loss.cpu().item()

def train_epoch(train_generator, model, loss_function, optimizer, scheduler, callback):
    """
    TODO
    """
    # создаем переменные для вычисления ошибки за эпоху
    epoch_loss = 0
    total = 0

    iterations = tqdm(enumerate(train_generator),
                      desc='batches',
                      leave=False,
                      total=math.ceil(train_generator.dataset.__len__()/train_generator.batch_size))
    iterations.set_postfix({'train batch loss': np.nan, 'lr': np.nan})

    # цикл по числу батчей
    for it, (batch_ids, batch_masks, batch_y) in iterations:
        # прогоняем батч через trainer
        batch_loss = train_on_batch(model, batch_ids.to('cuda'), batch_masks.to('cuda'),
                                    batch_y.to('cuda'), optimizer, scheduler, loss_function)
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        if callback is not None:
            model.eval()
            callback(model, batch_loss)
        iterations.set_postfix({'train batch loss': batch_loss, 'lr': current_lr})
        # обновляем ошибку за эпоху
        epoch_loss += batch_loss*len(batch_y)
        total += len(batch_y)
    # возвращаем ошибку за эпоху
    return epoch_loss/total

def trainer(n_epochs,
            batch_size,
            train_dataset,
            model,
            loss_function=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
            scheduler_params={"lr_lambda": lambda epoch: 0.95 ** epoch},
            lr = 2e-5,
            rs = 42,
            callback = None):
    """
    TODO
    """
    # инициализируем оптимизатор
    optima = optimizer(model.parameters(), lr=lr)
    # инициализиурем scheduler
    if scheduler != None:
        scheduler_params['optimizer'] = optima
        scheduler = scheduler(**scheduler_params)
    # инициализируем шкалу прогресса (по количеству эпох)
    iterations = tqdm(range(n_epochs), desc='epochs', leave=True)
    # задаем место для вывода ошибки каждую эпоху
    iterations.set_postfix({'train epoch loss': np.nan})
    # генерируем батчи тестовой выборки
    #test_generator = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size,
    #                                                  shuffle=True,
    #                                                  generator=torch.Generator().manual_seed(rs))
    # цикл по эпохам
    for it in iterations:
        # генерируем батчи
        batch_generator = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                      shuffle=True,
                                                      generator=torch.Generator().manual_seed(rs))
        # прогоняем батчи через trainer
        epoch_loss = train_epoch(train_generator=batch_generator,
                                 model=model,
                                 loss_function=loss_function,
                                 optimizer=optima,
                                 scheduler=scheduler,
                                 callback=callback)
        # выводим ошибку в прогрессе
        iterations.set_postfix({'train epoch loss': epoch_loss})
    return model