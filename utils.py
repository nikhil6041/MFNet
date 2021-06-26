import numpy as np
import torch
import visdom
from path import Path

def train_epoch(
  model, 
  data_loader, 
  criterion, 
  optimizer, 
  device, 
  n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    print(f'Doing training on {n_examples} samples')

    for batch_idx , (b_images,b_labels) in enumerate(data_loader):

        if batch_idx%10 == 0:
            print(f' Processing batch {batch_idx+1}/{len(data_loader)} ')

        b_images = b_images.to(device)
        b_labels = b_labels.to(device)


        outputs = model(b_images)
        _, b_preds = torch.max(outputs, 1)
        
        loss = criterion(outputs,b_labels)

        correct_predictions += torch.sum(b_preds == b_labels)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    accuracy = correct_predictions.double() / n_examples
    loss = round(np.mean(losses),2)

    return accuracy , loss

def eval_model(
    model, 
    data_loader,
    criterion,
    device,
    n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    print(f'Doing validation on {n_examples} samples')

    with torch.no_grad():

        for batch_idx , (b_images,b_labels) in enumerate(data_loader):

            if batch_idx%10 == 0:
                print(f' Processing batch {batch_idx+1}/{len(data_loader)} ')

            b_images = b_images.to(device)
            b_labels = b_labels.to(device)


            outputs = model(b_images)
            _, b_preds = torch.max(outputs, 1)

            # print(b_preds.size())
            # print(b_labels.size())
            
            loss = criterion(outputs,b_labels)

            correct_predictions += torch.sum(b_preds == b_labels)

            losses.append(loss.item())
    
    accuracy = correct_predictions.double() / n_examples
    loss = round(np.mean(losses),2)
    
    return accuracy , loss

def train_model(
    model,
    train_data_loader,
    val_data_loader, 
    train_dataset_size,
    val_dataset_size,
    optimizer,
    criterion,
    scheduler,
    device, 
    n_epochs=3):

    history = defaultdict(list)

    best_accuracy = 0
    criterion.to(device)

    for epoch in range(n_epochs):

        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
                                    model, 
                                    train_data_loader, 
                                    criterion, 
                                    optimizer, 
                                    device, 
                                    train_dataset_size
                                )

        print("Train loss {:.2f} accuracy {:.2f}".format(train_loss,train_acc))

        val_acc, val_loss = eval_model(
                                    model, 
                                    val_data_loader, 
                                    criterion, 
                                    device, 
                                    val_dataset_size

                            )

        print("Validation  loss {:.2f} accuracy {:.2f}".format(val_loss,val_acc))
        
        print()

        scheduler.step(val_loss)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    print(f'Best val accuracy: {best_accuracy}')

    model.load_state_dict(torch.load('best_model_state.bin'))

    return model, history

def visualize_images(path,n_samples):
    '''
        path: expects list of paths to images or a single image's path
    '''
    cnt = 0
    for root,dirs,files in os.walk(path):
        for fname in files:
            if cnt == n_samples:
                return
            path = os.path.join(root,fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            cv2_imshow(img)
          
            cnt += 1

def plot_training_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

  ax1.plot(history['train_loss'], label='train loss')
  ax1.plot(history['val_loss'], label='validation loss')

  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.legend()
  ax1.set_ylabel('Loss')
  ax1.set_xlabel('Epoch')

  ax2.plot(history['train_acc'], label='train accuracy')
  ax2.plot(history['val_acc'], label='validation accuracy')

  ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2.set_ylim([-0.05, 1.05])
  ax2.legend()

  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')

  fig.suptitle('Training history')

def show_confusion_matrix(confusion_matrix, class_names):

  cm = confusion_matrix.copy()

  cell_counts = cm.flatten()

  cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]

  row_percentages = ["{0:.2f}".format(value) for value in cm_row_norm.flatten()]

  cell_labels = [f"{cnt}\n{per}" for cnt, per in zip(cell_counts, row_percentages)]
  cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])

  df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)

  hmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Sign')
  plt.xlabel('Predicted Sign');


def write_csv(file, newrow):
    with open(file, mode='a') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(newrow)

class ModelSaver():

    def __init__(self):
        self._previous_acc = 0.
        self._current_acc = 0.

    @property
    def previous_acc(self):
        return self._previous_acc

    @property
    def current_acc(self):
        return self._current_acc

    @current_acc.setter
    def current_acc(self, value):
        self._current_acc = value

    @previous_acc.setter
    def previous_acc(self, value):
        self._previous_acc = value

    def __set_accuracy(self, accuracy):
        self.previous_acc, self.current_acc = self.current_acc, accuracy

    def save_if_best(self, accuracy, state):
        if accuracy > self.current_acc:
            self.__set_accuracy(accuracy)
            torch.save(state, 'log/best_state.pth')


def create_if_not_exist(path):
    path = Path(path)
    if not path.exists(): path.touch()


def init_log_just_created(path):
    create_if_not_exist(path)
    with open(path, 'r') as f:
        if len(f.readlines()) <= 0:
            init_log_line(path)


def init_log_line(path):
    with open(path, 'w') as f:
        f.write('time,epoch,acc,loss,layers,bs,lr\n')


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.viz.check_connection()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name],
                          name=split_name, update='append')

  

