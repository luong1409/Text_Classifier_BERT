from model import BertClassifier
import argparse
from utils import preprocess, train
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lr', help='learning rate', default=1e-6)
parser.add_argument('-e', '--epochs', help="number of epochs", default=5)


if __name__ == "__main__":
    train_df, val_df, test_df = preprocess()
    
    args = parser.parse_args()
    model = BertClassifier()
    train(model=model, train_data=train_df, 
          val_data=val_df, learning_rate=args.lr, 
          epochs=args.epochs)

    state = {
        'state_dict': model.state_dict()
    }
    save_path = 'checkpoint.std'
    
    torch.save(state, save_path)