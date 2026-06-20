import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers, models

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'rps_data')
TRAIN_DIR = os.path.join(DATA_DIR, 'rps')
TEST_DIR = os.path.join(DATA_DIR, 'rps-test-set')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULT_DIR, exist_ok=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)


def load_data_fast(root_dir, img_size=(48, 48), max_per_class=None):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    images = []
    labels = []

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if max_per_class is not None:
            files = files[:max_per_class]
        for fname in files:
            img_path = os.path.join(cls_dir, fname)
            image = Image.open(img_path).convert('RGB')
            image = image.resize(img_size, Image.BILINEAR)
            arr = np.array(image, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            images.append(arr)
            labels.append(class_to_idx[cls])

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int32)
    return images, labels, classes, class_to_idx


def step1():
    print('=' * 60)
    print('实验步骤1: 验证数据集')
    print('=' * 60)

    total_train = 0
    for category in sorted(os.listdir(TRAIN_DIR)):
        cat_dir = os.path.join(TRAIN_DIR, category)
        if os.path.isdir(cat_dir):
            count = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f'  训练集 {category}: {count} 张')
            total_train += count
    print(f'  训练集总计: {total_train} 张')

    total_test = 0
    for category in sorted(os.listdir(TEST_DIR)):
        cat_dir = os.path.join(TEST_DIR, category)
        if os.path.isdir(cat_dir):
            count = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f'  测试集 {category}: {count} 张')
            total_test += count
    print(f'  测试集总计: {total_test} 张')

    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    categories = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    for row, category in enumerate(categories):
        cat_dir = os.path.join(TRAIN_DIR, category)
        images = sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:5]
        for col, img_file in enumerate(images):
            img = Image.open(os.path.join(cat_dir, img_file))
            axes[row, col].imshow(img)
            axes[row, col].set_title(category)
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, '01_sample_images.png'), dpi=80)
    plt.close()
    print('  示例图片已保存')
    return categories


def step2(categories):
    print()
    print('=' * 60)
    print('实验步骤2: 使用Keras构建并训练模型')
    print('=' * 60)

    img_size = 48
    batch_size = 64
    epochs = 5
    num_classes = len(categories)
    max_train = 200
    max_test = 80

    print(f'图片尺寸: {img_size}x{img_size}, Batch: {batch_size}, Epochs: {epochs}')
    print(f'每类训练样本: {max_train}, 每类测试样本: {max_test}')

    print('加载数据...', flush=True)
    X_train, y_train, class_names, _ = load_data_fast(TRAIN_DIR, (img_size, img_size), max_train)
    X_test, y_test, _, _ = load_data_fast(TEST_DIR, (img_size, img_size), max_test)
    print(f'  训练样本: {len(X_train)}, 测试样本: {len(X_test)}')

    print('定义模型架构...')
    model = models.Sequential([
        layers.Input(shape=(3, img_size, img_size)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first'),
        layers.MaxPooling2D((2, 2), data_format='channels_first'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first'),
        layers.MaxPooling2D((2, 2), data_format='channels_first'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first'),
        layers.MaxPooling2D((2, 2), data_format='channels_first'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary(print_fn=lambda x: print('  ' + x))

    print('编译模型...')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('开始训练...', flush=True)
    num_train = len(X_train)
    num_test = len(X_test)

    train_acc_list, train_loss_list = [], []
    val_acc_list, val_loss_list = [], []

    start_time = time.time()
    for epoch in range(epochs):
        indices = np.random.permutation(num_train)
        X_shuf = X_train[indices]
        y_shuf = y_train[indices]

        train_losses, train_correct, train_total = [], 0, 0
        for i in range(0, num_train, batch_size):
            bx = X_shuf[i:i+batch_size]
            by = y_shuf[i:i+batch_size]
            loss, _ = model.train_on_batch(bx, by)
            train_losses.append(loss)
            preds = to_numpy(model(bx, training=False))
            train_correct += np.sum(np.argmax(preds, axis=1) == by)
            train_total += len(by)

        avg_train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total

        val_losses, val_correct, val_total = [], 0, 0
        for i in range(0, num_test, batch_size):
            bx = X_test[i:i+batch_size]
            by = y_test[i:i+batch_size]
            results = model.evaluate(bx, by, verbose=0, return_dict=True)
            val_losses.append(results['loss'])
            preds = to_numpy(model(bx, training=False))
            val_correct += np.sum(np.argmax(preds, axis=1) == by)
            val_total += len(by)

        avg_val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        train_acc_list.append(train_acc)
        train_loss_list.append(avg_train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs}: loss={avg_train_loss:.4f}, acc={train_acc:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}', flush=True)

    elapsed = time.time() - start_time
    print(f'训练完成! 总耗时: {elapsed:.1f}秒')

    class History:
        def __init__(self):
            self.history = {'accuracy': train_acc_list, 'loss': train_loss_list,
                            'val_accuracy': val_acc_list, 'val_loss': val_loss_list}

    model.save(os.path.join(RESULT_DIR, 'rps_model.keras'))
    return model, History(), X_test, y_test, class_names


def step3(model, history, X_test, y_test, class_names):
    print()
    print('=' * 60)
    print('实验步骤3: 验证模型并绘制性能图')
    print('=' * 60)

    print('绘制训练曲线...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], 'b-o', label='Train')
    axes[0].plot(history.history['val_accuracy'], 'r-o', label='Validation')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(history.history['loss'], 'b-o', label='Train')
    axes[1].plot(history.history['val_loss'], 'r-o', label='Validation')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, '02_training_curves.png'), dpi=80)
    plt.close()
    print('  训练曲线已保存')

    print('评估测试集...')
    batch_size = 64
    num_test = len(X_test)
    test_correct, test_total, test_losses = 0, 0, []
    all_preds, all_labels = [], []

    for i in range(0, num_test, batch_size):
        bx = X_test[i:i+batch_size]
        by = y_test[i:i+batch_size]
        results = model.evaluate(bx, by, verbose=0, return_dict=True)
        test_losses.append(results['loss'])
        preds = to_numpy(model(bx, training=False))
        pred_labels = np.argmax(preds, axis=1)
        test_correct += np.sum(pred_labels == by)
        test_total += len(by)
        all_preds.extend(pred_labels)
        all_labels.extend(by)

    test_acc = test_correct / test_total
    test_loss = np.mean(test_losses)
    print(f'  测试集准确率: {test_acc:.4f} ({test_acc*100:.1f}%)')

    print('绘制预测示例...')
    num_imgs = 12
    idx = np.random.choice(num_test, num_imgs, replace=False)
    sample_X, sample_y = X_test[idx], y_test[idx]
    preds = to_numpy(model(sample_X, training=False))
    pred_labels = np.argmax(preds, axis=1)

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    for i in range(num_imgs):
        ax = axes[i // cols, i % cols]
        img_disp = np.transpose(sample_X[i], (1, 2, 0))
        ax.imshow(img_disp)
        pred_c = class_names[pred_labels[i]]
        true_c = class_names[sample_y[i]]
        conf = preds[i][pred_labels[i]] * 100
        correct = pred_c == true_c
        color = 'green' if correct else 'red'
        ax.set_title(f'Pred:{pred_c}\nTrue:{true_c}\nConf:{conf:.0f}%', fontsize=9, color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, '03_predictions.png'), dpi=80)
    plt.close()
    print('  预测示例已保存')

    print('绘制混淆矩阵...')
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, '04_confusion_matrix.png'), dpi=80)
    plt.close()
    print('  混淆矩阵已保存')

    print('分类报告:')
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_str)

    report_path = os.path.join(RESULT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write('RPS Model Experiment Report\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)\n')
        f.write(f'Test loss: {test_loss:.4f}\n\n')
        train_acc = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        f.write(f'Train accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)\n')
        f.write(f'Train loss: {train_loss:.4f}\n\n')
        f.write(f'Validation accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)\n')
        f.write(f'Validation loss: {val_loss:.4f}\n\n')
        f.write('Classification Report:\n')
        f.write(report_str)
        f.write('\n\nPer-Epoch Details:\n')
        for epoch in range(len(history.history['accuracy'])):
            f.write(f'Epoch {epoch+1:2d}: acc={history.history["accuracy"][epoch]:.4f}, '
                    f'loss={history.history["loss"][epoch]:.4f}, '
                    f'val_acc={history.history["val_accuracy"][epoch]:.4f}, '
                    f'val_loss={history.history["val_loss"][epoch]:.4f}\n')
    print(f'  报告已保存: {report_path}')


def main():
    print('=' * 60)
    print('TensorFlow/Keras 石头剪刀布模型实验')
    print('=' * 60)

    try:
        categories = step1()
        model, history, X_test, y_test, class_names = step2(categories)
        step3(model, history, X_test, y_test, class_names)
    except Exception as e:
        import traceback
        print(f'\n错误: {e}')
        traceback.print_exc()
        return

    print()
    print('=' * 60)
    print(f'实验完成! 所有结果已保存到 {RESULT_DIR}/ 目录')
    print('=' * 60)


if __name__ == '__main__':
    main()
