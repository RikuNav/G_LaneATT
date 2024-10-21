import os
import re
import torch

def load_last_train_state(model, optimizer, scheduler):
    """
        Load the last training state from the checkpoint files.

        Args:
            model: The model to be loaded.
            optimizer: The optimizer to be loaded.
            scheduler: The scheduler to be loaded.
        
        Returns:
            epoch: The epoch of the last checkpoint.
            model: The model loaded from the last checkpoint.
            optimizer: The optimizer loaded from the last checkpoint.
            scheduler: The scheduler loaded from the last checkpoint.
    """
    
    train_state_path, epoch = get_last_checkpoint()
    train_state = torch.load(train_state_path)
    model.load_state_dict(train_state['model'])
    optimizer.load_state_dict(train_state['optimizer'])
    scheduler.load_state_dict(train_state['scheduler'])

    return epoch, model, optimizer, scheduler

def get_last_checkpoint():
    """
        Get the epoch of the last checkpoint.

        Returns:
            The epoch of the last checkpoint.
    """
    
    # Generate the pattern to match the checkpoint files and a list of all the checkpoint files
    pattern = re.compile('model_(\\d+)_(\\d+).pt')
    checkpoints = os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'))
    checkpoints = [ckpt for ckpt in checkpoints if re.match(pattern, ckpt) is not None]

    # Get last checkpoint epoch
    latest_checkpoint_path = sorted(checkpoints, reverse=True, key=lambda name : name.split('_')[1])[0]
    epoch = latest_checkpoint_path.split('_')[2].rstrip('.pt')

    return latest_checkpoint_path, int(epoch)

if __name__ == '__main__':
    pass