import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Concatenate,
    Input, GlobalAveragePooling2D, Activation, Multiply,
    Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
)
import math


# ============================================================
#  Config
# ============================================================
IMG_SIZE       = 224
BATCH_SIZE     = 16
PHASE1_EPOCHS  = 30
PHASE2_EPOCHS  = 80
PHASE1_LR      = 1e-3
PHASE2_LR      = 1e-5
LABEL_SMOOTH   = 0.1
UNFREEZE_FRACTION = 0.3


# ============================================================
#  Preprocessing
# ============================================================
def preprocess_input(x):
    """Scale pixel values to [-1, 1]."""
    return x / 127.5 - 1.0


# ============================================================
#  Cosine Annealing LR Schedule
# ============================================================
def cosine_lr_schedule(epoch, lr, total_epochs=PHASE2_EPOCHS, min_lr=1e-7):
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

# ============================================================
#  Cutout Augmentation
# ============================================================
def cutout(image, mask_size=50):
    h, w, c = image.shape
    top  = np.random.randint(0, h - mask_size) if h > mask_size else 0
    left = np.random.randint(0, w - mask_size) if w > mask_size else 0
    image[top:top+mask_size, left:left+mask_size, :] = 0.0
    return image


def augment_with_cutout(image):
    image = preprocess_input(image)
    if np.random.rand() < 0.5:
        image = cutout(image, mask_size=int(IMG_SIZE * 0.15))
    return image


# ============================================================
#  Dataset Loader  (all body parts → label = body-part name)
# ============================================================
def load_path(path):
    """
    Walk:  Dataset/{train_valid,test}/<Part>/patient/study_pos|neg/img
    Label is the body-part name (Elbow / Hand / Shoulder).
    """
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        for body in os.listdir(folder_path):
            path_p = os.path.join(folder_path, body)
            if not os.path.isdir(path_p):
                continue
            for id_p in os.listdir(path_p):
                path_id = os.path.join(path_p, id_p)
                if not os.path.isdir(path_id):
                    continue
                for lab in os.listdir(path_id):
                    path_l = os.path.join(path_id, lab)
                    if not os.path.isdir(path_l):
                        continue
                    for img in os.listdir(path_l):
                        dataset.append({
                            'label': body,
                            'image_path': os.path.join(path_l, img)
                        })
    return dataset


# ============================================================
#  Channel Attention (SE Block)
# ============================================================
def se_block(x, ratio=16, name_prefix='se'):
    channels = x.shape[-1]
    se = GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    se = Dense(channels // ratio, activation='relu', name=f'{name_prefix}_fc1')(se)
    se = Dense(channels, activation='sigmoid', name=f'{name_prefix}_fc2')(se)
    se = Reshape((1, 1, channels), name=f'{name_prefix}_reshape')(se)
    return Multiply(name=f'{name_prefix}_scale')([x, se])


# ============================================================
#  Ensemble Model Builder  (UPGRADED — same arch as fracture)
# ============================================================
def build_ensemble_model(num_classes):
    """
    Upgraded Ensemble: MobileNetV2 + DenseNet121 + InceptionV3
    with SE attention and deeper head.
    """
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_image')

    # Backbone 1 — MobileNetV2
    mob = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights='imagenet', pooling=None)
    mob._name = 'mobilenetv2'
    mob.trainable = False
    mob_out  = mob(inp)
    mob_out  = se_block(mob_out, ratio=16, name_prefix='mob_se')
    mob_feat = Activation('linear', name='mob_conv_output')(mob_out)
    mob_pool = GlobalAveragePooling2D(name='mob_gap')(mob_feat)

    # Backbone 2 — DenseNet121
    dn = tf.keras.applications.DenseNet121(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights='imagenet', pooling=None)
    dn._name = 'densenet121'
    dn.trainable = False
    dn_out  = dn(inp)
    dn_out  = se_block(dn_out, ratio=16, name_prefix='dense_se')
    dn_feat = Activation('linear', name='dense_conv_output')(dn_out)
    dn_pool = GlobalAveragePooling2D(name='dense_gap')(dn_feat)

    # Backbone 3 — InceptionV3
    inc = tf.keras.applications.InceptionV3(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights='imagenet', pooling=None)
    inc._name = 'inceptionv3'
    inc.trainable = False
    inc_out  = inc(inp)
    inc_out  = se_block(inc_out, ratio=16, name_prefix='inc_se')
    inc_feat = Activation('linear', name='inc_conv_output')(inc_out)
    inc_pool = GlobalAveragePooling2D(name='inc_gap')(inc_feat)

    # Merge & classify
    x = Concatenate(name='ensemble_concat')([mob_pool, dn_pool, inc_pool])
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(512, activation='relu', name='fc_1')(x)
    x = Dropout(0.4, name='drop_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dense(256, activation='relu', name='fc_2')(x)
    x = Dropout(0.3, name='drop_2')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Dense(128, activation='relu', name='fc_3')(x)
    x = Dropout(0.2, name='drop_3')(x)
    out = Dense(num_classes, activation='softmax', name='predictions')(x)

    return Model(inputs=inp, outputs=out, name='ensemble_model')


# ============================================================
#  Unfreeze top N% of a backbone
# ============================================================
def unfreeze_top_layers(model, backbone_name, fraction=UNFREEZE_FRACTION):
    for layer in model.layers:
        if layer.name == backbone_name:
            sub = layer
            break
    else:
        print(f"  ⚠  Backbone '{backbone_name}' not found — skipping unfreeze")
        return

    total = len(sub.layers)
    freeze_up_to = int(total * (1 - fraction))
    for i, layer in enumerate(sub.layers):
        layer.trainable = i >= freeze_up_to

    trainable_count = sum(1 for l in sub.layers if l.trainable)
    print(f"  ✔ {backbone_name}: unfroze {trainable_count}/{total} layers "
          f"(top {fraction*100:.0f} %)")


# ============================================================
#  Plot helpers
# ============================================================
def save_plots(h1, h2, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    acc     = h1.history['accuracy']      + h2.history['accuracy']
    val_acc = h1.history['val_accuracy']   + h2.history['val_accuracy']
    loss    = h1.history['loss']           + h2.history['loss']
    val_loss= h1.history['val_loss']       + h2.history['val_loss']
    phase_boundary = len(h1.history['accuracy'])

    plt.figure(figsize=(12, 6))
    plt.plot(acc, label='Train', linewidth=2)
    plt.plot(val_acc, label='Validation', linewidth=2)
    plt.axvline(x=phase_boundary, color='red', linestyle='--', alpha=0.6,
                label='Fine-tune starts')
    plt.title('Ensemble V2 Body-Part Accuracy', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Ensemble_BodyPart_Accuracy.jpeg'), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(loss, label='Train', linewidth=2)
    plt.plot(val_loss, label='Validation', linewidth=2)
    plt.axvline(x=phase_boundary, color='red', linestyle='--', alpha=0.6,
                label='Fine-tune starts')
    plt.title('Ensemble V2 Body-Part Loss', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Ensemble_BodyPart_Loss.jpeg'), dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_labels, plot_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix — Body Parts', fontsize=14)
    plt.colorbar()
    ticks = np.arange(len(class_labels))
    plt.xticks(ticks, class_labels, fontsize=11)
    plt.yticks(ticks, class_labels, fontsize=11)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=14)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Ensemble_BodyPart_CM.jpeg'), dpi=150)
    plt.close()


# ============================================================
#  Main training script  —  TWO-PHASE
# ============================================================
if __name__ == '__main__':
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(THIS_FOLDER, 'Dataset')
    data = load_path(image_dir)

    filepaths = pd.Series([r['image_path'] for r in data], name='Filepath').astype(str)
    labels = pd.Series([r['label'] for r in data], name='Label')
    images = pd.concat([filepaths, labels], axis=1)

    Labels = ["Elbow", "Hand", "Shoulder"]
    print(f"\n{'='*60}\n  Body-Part Classification  |  Total: {len(images)}")
    print(f"  {dict(images['Label'].value_counts())}\n{'='*60}\n")

    train_df, test_df = train_test_split(
        images, train_size=0.9, shuffle=True, random_state=1)

    # --- generators with stronger augmentation ---
    aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        channel_shift_range=20.0,
        fill_mode='reflect',
        preprocessing_function=augment_with_cutout,
        validation_split=0.2)

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input)

    flow_args = dict(x_col='Filepath', y_col='Label',
                     target_size=(IMG_SIZE, IMG_SIZE),
                     color_mode='rgb', class_mode='categorical',
                     batch_size=BATCH_SIZE, seed=42)

    train_it = aug_gen.flow_from_dataframe(train_df, shuffle=True,
                                           subset='training', **flow_args)
    val_it   = aug_gen.flow_from_dataframe(train_df, shuffle=True,
                                           subset='validation', **flow_args)
    test_it  = test_gen.flow_from_dataframe(test_df, shuffle=False, **flow_args)

    # --- class weights ---
    total = len(train_it.classes)
    class_counts = np.bincount(train_it.classes)
    class_weight = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    print(f"  Class weights: {class_weight}")

    weights_dir = os.path.join(THIS_FOLDER, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────
    #  PHASE 1: Head warm-up
    # ────────────────────────────────────────────────────────
    print(f"\n  ▸ PHASE 1 — Head warm-up ({PHASE1_EPOCHS} epochs, lr={PHASE1_LR})\n")
    model = build_ensemble_model(num_classes=len(Labels))
    model.compile(
        optimizer=Adam(learning_rate=PHASE1_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy'])
    model.summary()

    p1_callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]

    h1 = model.fit(train_it, validation_data=val_it,
                   epochs=PHASE1_EPOCHS, callbacks=p1_callbacks,
                   class_weight=class_weight)

    p1_acc = max(h1.history['val_accuracy'])
    print(f"\n  ✔ Phase 1 best val_accuracy: {p1_acc*100:.2f} %")

    # ────────────────────────────────────────────────────────
    #  PHASE 2: Fine-tuning
    # ────────────────────────────────────────────────────────
    print(f"\n  ▸ PHASE 2 — Fine-tuning ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR})\n")

    unfreeze_top_layers(model, 'mobilenetv2',     UNFREEZE_FRACTION)
    unfreeze_top_layers(model, 'densenet121',     UNFREEZE_FRACTION)
    unfreeze_top_layers(model, 'inceptionv3',     UNFREEZE_FRACTION)

    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy'])

    p2_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(weights_dir, 'Ensemble_BodyParts_best.h5'),
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        LearningRateScheduler(cosine_lr_schedule, verbose=1),
    ]

    h2 = model.fit(train_it, validation_data=val_it,
                   epochs=PHASE2_EPOCHS, callbacks=p2_callbacks,
                   class_weight=class_weight)

    # --- save ---
    final_path = os.path.join(weights_dir, 'Ensemble_BodyParts.h5')
    model.save(final_path)
    print(f"\n  ✔ Model saved → {final_path}")

    loss, acc = model.evaluate(test_it, verbose=0)
    print(f"  Test Loss: {loss:.4f}  |  Test Accuracy: {np.round(acc*100, 2)}%")

    y_pred = np.argmax(model.predict(test_it), axis=1)
    y_true = test_it.classes
    labels_list = list(test_it.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=labels_list))

    plot_dir = os.path.join(THIS_FOLDER, 'plots')
    save_plots(h1, h2, plot_dir)
    save_confusion_matrix(y_true, y_pred, labels_list, plot_dir)
    print(f"  ✔ Plots saved → {plot_dir}")
    print("\n✅ Ensemble body-part model trained successfully!")


