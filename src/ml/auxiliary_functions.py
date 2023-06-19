import gc
import os
from torch import cuda
from transformers import AutoTokenizer, AutoModel
from tensorboard.backend.event_processing import event_accumulator


def clear_gpu_memory():
    """
    TODO
    """
    cuda.empty_cache()
    gc.collect()


def load_model(model_checkpoint, n_labels):
    '''
    Function to load a Hugging Face model
    * Input:
        - model_checkpoint: checkpoint of a model to be loaded
        - n_labels: number of labels (targets) in data
    * Output: Object of a model
    '''
    model = AutoModel.from_pretrained(model_checkpoint, num_labels=n_labels).to('cpu')
    return model

def load_tokenizer(model_checkpoint, use_fast):
    '''
    Function to load a Hugging Face tokenizer
    * Input:
        - model_checkpoint: checkpoint of a model's tokenizer to be loaded
    * Output: Object of a tokenizer
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=use_fast)
    return tokenizer

def tokenize_data(data, tokenizer, dataset_cls, max_length=100):
    '''
    Function to tokenizer text data
    * Input:
        - data: data in DataFrame format with at least ['text','label'] columns
        - tokenizer: Object of a tokenizer
        - max_length: max length of a single text (only the beginning with max_length length will remain in
                      case of a bigger text)
        - dataset_cls: Dataset object to store data
    * Output: data in a form of dataset_cls object
    '''
    output = tokenizer(list(data['text'].values), padding=True,
                       truncation=True, max_length=max_length, return_tensors='pt')
    output['input_ids'] = output['input_ids']
    output['attention_mask'] = output['attention_mask']
    dataset = dataset_cls(output, data['label'].values)
    return dataset


def get_text_log(path, filename):
    """
    TODO
    """
    ea = event_accumulator.EventAccumulator(os.path.join(path, filename))
    ea.Reload()
    tag = ea.Tags()['tensors'][0]
    logs = ea.Tensors(tag)
    steps = [log.step for log in logs]
    logs = [log.tensor_proto.string_val[0].decode() for log in logs]
    return steps, logs


def get_scalar_log(path, filename):
    """
    TODO
    """
    ea = event_accumulator.EventAccumulator(os.path.join(path, filename))
    ea.Reload()
    tag = ea.Tags()['scalars'][0]
    logs = ea.Scalars(tag)
    steps = [log.step for log in logs]
    logs = [log.value for log in logs]
    return steps, logs