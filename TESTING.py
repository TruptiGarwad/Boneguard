import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk  # Import PIL for image rendering
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np

# Load the pre-trained VGG19 model
#model = load_model('model_vgg19.h5')

# Create the main application window
root = tk.Tk()
root.title("Image Classifier")

# Function to classify the selected image
def classify_image(self):
    global img_label

    # Open a file dialog for the user to select an image
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if not file_path:
        result_label.config(text="No image selected.")
        return

    # Load and preprocess the selected image
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)  # Apply VGG19-specific preprocessing

    # Expand dimensions to match the model's input shape
    x = np.expand_dims(x, axis=0)

    # Make predictions
    classes = model.predict(x)

    # Assuming class 0 is malignant and class 1 is normal
    malignant_prob = classes[0, 0]
    normal_prob = classes[0, 1]

    # Determine the class based on probabilities
    if malignant_prob > normal_prob:
        result_label.config(text='Predicted Class: Malignant', fg='red')
    else:
        result_label.config(text='Predicted Class: Normal', fg='green')

    # Render the selected image in the GUI
    img_pil = Image.open(file_path)
    img_render = ImageTk.PhotoImage(img_pil)
    img_label.config(image=img_render)
    img_label.image = img_render

# Create a label for displaying the result
result_label = Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a label for displaying the selected image
img_label = Label(root)
img_label.pack()

# Create a classify button
classify_button = Button(root, text="Classify Image", command=classify_image)
classify_button.pack()

# Run the Tkinter main loop
root.mainloop()
