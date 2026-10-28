from dataset import ClassificationDataset
from model import ClassifierModel

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torch.utils.tensorboard import SummaryWriter

from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classification Trainning Script')
    parser.add_argument('--image-size', type=int, default=64, help='Image size')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num-workers')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--data-path', type=str, default='D:/AI/datasets/animals10', help='Dataset path')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoints Directory')
    parser.add_argument('--tensorboard-dir', type=str, default='./logs', help='TensorBoard Directory')
    
    return parser.parse_args()

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    #color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_matrix', figure, epoch)
    plt.close(figure)

def train(args):
    #Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Ensure directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    #Data transformations
    TRANSFORM = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])

    #Train params and Val params
    TRAIN_PARAMS = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True
    }

    VAL_PARAMS = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True
    }

    #Dataset and DataLoader
    train_set = ClassificationDataset(root=args.data_path, split='train', transform=TRANSFORM)
    val_set = ClassificationDataset(root=args.data_path, split='val', transform=TRANSFORM)

    train_loader = DataLoader(dataset=train_set, **TRAIN_PARAMS)
    val_loader = DataLoader(dataset=val_set, **VAL_PARAMS)
        
    #Model, Loss, Optimizer
    model = ClassifierModel(num_classes=len(train_set.classes), input_shape=(3, args.image_size, args.image_size)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #Tensorboard writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    #Load checkpoint if resuming
    best_val_loss = np.inf
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'last_model.pt'), map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    print(f'::Train from epoch:{start_epoch}')

    #Train & Val
    for epoch in range(start_epoch, args.epochs):
        #Train
        model.train()
        train_loss = 0
        train_progress_bar = tqdm(train_loader, colour='magenta')
        for iter, (images, labels) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            # print(loss.item())

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update progress bar
            train_progress_bar.set_description(f'Epoch: {epoch+1}/{args.epochs}, Loss:{loss.item():.4f}')
            # writer.add_scalar(tag='Train/Loss', scalar_value=train_loss, global_step=(epoch*len(train_loader)+iter))

        #Logs Train
        train_loss /= len(train_loader)
        writer.add_scalar(tag='Train/Loss', scalar_value=train_loss, global_step=epoch)

        #Validation
        model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        val_progress_bar = tqdm(val_loader, colour='green', desc='Validation::')
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        #Calculate metrics and log to Tensorboard
        val_loss /= len(val_loader)
        val_acc = accuracy_score(y_true=all_labels, y_pred=all_predictions)

        writer.add_scalar(tag='Val/Loss', scalar_value=val_loss, global_step=epoch)
        writer.add_scalar(tag='Val/Accuracy', scalar_value=val_acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(y_true=all_labels, y_pred=all_predictions), train_set.classes, epoch)


        #Save checkpoint
        checkpoint = {
            'classes': train_set.classes,
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f'::Model saved with validation loss:{val_loss:.4f}, accuracy:{(val_acc*100):.2f}%\n')
        else:
            print(f'Val_loss::{val_loss:.4f}, val_accuracy:{(val_acc*100):.2f}%\n')
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'last_model.pt'))

if __name__ == '__main__':
    args = parse_arguments()
    train(args)