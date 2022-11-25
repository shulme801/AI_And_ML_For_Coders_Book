import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
       if (logs.get('accuracy')>0.95):
        print("\nReached 95% accuracy so cancelling training!")
        self.model.stop_training = True

def computer_vision(num_epochs):
   callbacks = myCallback()
   mnist = tf.keras.datasets.fashion_mnist
   (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
   training_images=training_images/255.0
   test_images=test_images/255.0
   model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    computer_vision(50)
    exit()
# That's all folks!
