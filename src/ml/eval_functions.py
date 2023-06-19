import math
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report

def evaluate(dataset,
             batch_size,
             model,
             metric=None):
    """
    TODO
    """
    model.eval()
    y_pred = []
    y_true = []
    generator = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                      shuffle=True)
    iterations = tqdm(enumerate(generator),
                      desc='Evaluation',
                      leave=False,
                      total=math.ceil(generator.dataset.__len__()/generator.batch_size))
    for it, (batch_ids, batch_masks, batch_y) in iterations:
        with torch.no_grad():
            logits = model(input_ids=batch_ids.to('cuda'), attention_mask=batch_masks.to('cuda')).cpu()
            y_pred += logits.argmax(dim=1).tolist()
            y_true += batch_y.tolist()
    output = {}
    if metric is not None:
        output['metric_score'] = metric(y_true, y_pred)
    output['classification_report'] = classification_report(y_true, y_pred)
    return output