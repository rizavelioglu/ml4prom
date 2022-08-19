from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def acc_per_class(y_test, y_test_pred):
    target_names = ["Lminus", "Lplus"]
    n_classes = len(target_names)

    class_correct = list(0 for i in range(n_classes))
    class_total = list(0 for i in range(n_classes))

    correct = np.equal(y_test, y_test_pred)
    # calculate test accuracy for each object class
    for i in range(len(y_test)):
        label = int(y_test[i])
        class_correct[label] += 1 if correct[i] else 0
        class_total[label] += 1

    for i in range(n_classes):
        print(f"Test Accuracy of {str(i)}:",
              f"{100 * class_correct[i] / class_total[i]:.2f}% ",
              f"({np.sum(class_correct[i])}/{np.sum(class_total[i])})  ",
              f"({target_names[i]})")

    print(f"\nTest Accuracy (Overall): {100. * np.sum(class_correct) / np.sum(class_total):.2f}%",
          f"({np.sum(class_correct)}/{np.sum(class_total)})\n",
          "*" * 50, sep='')

    return np.sum(class_correct) / np.sum(class_total)


def metrics_report(y_test, y_test_pred):

    target_names = ["Lminus", "Lplus"]

    print(f"F1-macro score: {f1_score(y_test, y_test_pred, average='macro') :.4f}",
          f"\nF1-weighted score : {f1_score(y_test, y_test_pred, average='weighted') :.4f}\n",
          "*" * 50, sep='')

    print("\n", classification_report(y_test,
                                      y_test_pred,
                                      target_names=target_names),
          "*" * 50, sep='')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    A function to plot confusion matrix. See the following website for more info:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
