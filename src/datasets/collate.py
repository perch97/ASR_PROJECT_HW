import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    
    batched_data = {}
    item_keys = dataset_items[0].keys()

    for key in item_keys:
        if key in {"spectrogram", "text_encoded"}:
            lengths = [entry[key].shape[-1] for entry in dataset_items]
            batched_data[f"{key}_length"] = torch.tensor(lengths)

            padded_data = pad_sequence(
                [entry[key].squeeze(0).T for entry in dataset_items],
                batch_first=True
            )
            batched_data[key] = padded_data
        else:
            batched_data[key] = [entry[key] for entry in dataset_items]

    if "spectrogram" in batched_data:
        batched_data["spectrogram"] = batched_data["spectrogram"].permute(0, 2, 1)

    return batched_data