import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]


    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        
        img_array = image.img_to_array(img)
        
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, 'Conv_1')
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite("gradcam_heatmap.png", superimposed_img)
        print("Heatmap saved as 'gradcam_heatmap.png'")


        # Display heatmap
        plt.matshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "lionandson.png"  
    classify_image(image_path)