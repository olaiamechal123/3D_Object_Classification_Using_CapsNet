from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, callbacks
import tensorflow as tf
import os
import numpy as np
import csv
from Functions import CapsNet3D, margin_loss, plot_log, load_voxel_data

def train(model, data, args):
    (x_train, y_train), (x_test, y_test) = data

    # Callbacks
    log = callbacks.CSVLogger(os.path.join(args['save_dir'], 'log.csv'))
    tb = callbacks.TensorBoard(log_dir=os.path.join(args['save_dir'], 'tensorboard-logs'),
                               histogram_freq=int(args['debug']))
    checkpoint = callbacks.ModelCheckpoint(os.path.join(args['save_dir'], 'weights-{epoch:02d}.weights.h5'),
                                        monitor='val_accuracy', save_best_only=True,
                                        save_weights_only=True, verbose=1)
    
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args['lr'] * (args['lr_decay'] ** epoch))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    
    # Compile the model
    model.compile(optimizer=optimizers.AdamW(learning_rate=args['lr']),
                  loss=[margin_loss],
                  metrics={'capsnet': 'accuracy'})

    # Train the model with the provided data
    model.fit(
        x=[x_train, y_train],  # Model inputs
        y=[y_train, x_train],  # Model targets
        batch_size=args['batch_size'],
        epochs=args['epochs'],
        validation_data=([x_test, y_test], [y_test, x_test]),
        callbacks=[log, tb, checkpoint, lr_decay, lr_schedule, early_stopping]
    )


    model.save_weights(os.path.join(args['save_dir'], 'trained_model.weights.h5'))
    print(f'Trained model saved to \'{args["save_dir"]}/trained_model.weights.h5\'')


    # Plot training log
    plot_log(os.path.join(args['save_dir'], 'log.csv'), show=True)

    return model


def Test(model, data):
    voxel_data, labels = data
    prediction = model.predict(voxel_data)
    return prediction
    

if __name__=="__main__":
    # Define the arguments
    args = {
        'epochs': 50,
        'batch_size': 4,
        'lr': 0.001,
        'lr_decay': 0.95,
        'lam_recon': 0.392,
        'routings': 3,
        'shift_fraction': 0.1,
        'debug': True,
        'save_dir': './result',
        'digit': 5,
        'train': False,
        'weights': r"C:\Users\SOUHAILA ELKADAOUI\Desktop\code (1)\code\Model\weights-47.weights.h5"
    }

    # Ensure the save directory exists
    os.makedirs(args['save_dir'], exist_ok=True)

    # Define the model (assumes you have a CapsNet function defined)
    model, eval_model = CapsNet3D(input_shape=(30, 30, 30, 1),
                                                n_class=2,
                                                routings=args['routings'])
    model.summary()




    # Train or test the model
    if args['train']:
        # Load the data (update the function name to match your implementation)
        (x_train, y_train), (x_test, y_test) = load_voxel_data(data_dir=r"/kaggle/input/intra-in3d/data/data")

        trained_model = train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

    else:
        model = eval_model
        model.load_weights(args['weights'])
        direct = r"C:\Users\SOUHAILA ELKADAOUI\Desktop\InternShip\test"
        voxel_data, labels = load_voxel_data(data_dir=direct, test=True)
        data = voxel_data, labels
        predictions = Test(model = model, data=data)
        predictions_decoded = np.argmax(predictions, axis=1)
        labels_decoded = np.argmax(labels, axis=1)

        # Display predictions and true labels
        for pred, true_label in zip(predictions_decoded, labels_decoded):
            print(f"Prediction: {pred}, True Label: {true_label}")
        import numpy as np
        print(np.array(predictions).sum(axis=1))
        print(predictions)
