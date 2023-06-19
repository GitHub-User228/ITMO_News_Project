import math
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report


class callback():
    """
    TODO
    """
    def __init__(self, writer, dataset, loss_function,
                 eval_steps = 3,
                 batch_size=128):
        self.step = 0
        self.writer = writer
        self.eval_steps = eval_steps
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.dataset = dataset
        self.pred = []
        self.real = []

    def forward(self, model, loss):
        """
        TODO
        """
        self.step += 1

        self.writer.add_scalars('LOSS', {'train': loss}, self.step)

        if self.step == 1:
            self.writer.add_graph(model,
                                  [self.dataset[0:1][0].to(model.device),
                                   self.dataset[0:1][1].to(model.device)])

        if (self.step % self.eval_steps == 0) or (self.step==1):
            #self.writer.add_graph(model, self.dataset[0][0].view(1,1,28,28).to(model.device))
            batch_generator = torch.utils.data.DataLoader(dataset = self.dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=True)
            iterations = tqdm(enumerate(batch_generator),
                              desc='Evaluation',
                              leave=False,
                              total=math.ceil(batch_generator.dataset.__len__()/batch_generator.batch_size))
            pred = []
            real = []
            test_loss = 0
            for it, (batch_ids, batch_masks, batch_y) in iterations:
                with torch.no_grad():
                    batch_ids = batch_ids.to(model.device)
                    batch_masks = batch_masks.to(model.device)
                    batch_y = batch_y.to(model.device)

                    logits = model(batch_ids, batch_masks)

                    test_loss += self.loss_function(logits, batch_y).cpu().item()*len(batch_y)

                    pred += logits.argmax(axis=1).cpu().tolist()
                    real += batch_y.cpu().tolist()
            self.pred.append(pred)
            self.real.append(real)

            test_loss /= len(self.dataset)

            self.writer.add_scalars('LOSS', {'test': test_loss}, self.step)

            self.writer.add_text('REPORT/test', str(classification_report(real, pred)), self.step)


    def __call__(self, model, loss):
        return self.forward(model,loss)