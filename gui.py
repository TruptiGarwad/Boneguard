from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog
import os
import numpy as np
from PIL import ImageFile                            
import imutils
# global variables
from PIL import ImageTk, Image
global rep
from numpy import load
from skimage.feature import hog
from skimage.measure import shannon_entropy
from skimage.filters import unsharp_mask

from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops






from skimage.color import rgb2gray
from tkinter import messagebox
from sklearn.datasets import load_files       
from glob import glob
from keras.preprocessing import image                  
from tkinter import Label


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg="skyblue")
        
        # changing the title of our master widget
        
        self.master.title("bone tumour detection")
        
        self.pack(fill=BOTH, expand=1)
        
        w = tk.Label(root, 
		 text="BONE TUMOUR DETECTION",
		 fg = "black",    
		 bg = "#654321",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=400, y=0)

        # creating a button instance
        quitButton = Button(self,command=self.load, text="LOAD IMAGE",bg="#999900",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=50, y=100,anchor="w")
        quitButton = Button(self,command=self.preprocess,text="preprocessing",bg="#999900",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=50,y=200,anchor="w")
        quitButton = Button(self,command=self.segmentation, text="segment",bg="#999900",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=50, y=300,anchor="w")
        quitButton = Button(self,command=self.classification,text="PREDICT",bg="#999900",activebackground="dark red",fg="#4C0099",width=20)
        quitButton.place(x=50, y=400,anchor="w")
        
        load = Image.open("logo.jfif")
        render = ImageTk.PhotoImage(load)

        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image1.image = render
        image1.place(x=400, y=50)

        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image2.image = render
        image2.place(x=400, y=250)
        
        
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image4.image = render
        image4.place(x=400, y=450)
        
        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image5.image = render
        image5.place(x=650, y=450)


        image6=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100)
        image6.image = render
        image6.place(x=850, y=450)
        
        
        

#       Functions

    def load(self, event=None):
        global rep
        rep = filedialog.askopenfilenames()
        img = cv2.imread(rep[0])
        
        #Input_img=img.copy()
        print(rep[0])
        self.from_array = Image.fromarray(cv2.resize(img,(150,150)))
        load = Image.open(rep[0])
        render = ImageTk.PhotoImage(load.resize((150,150)))
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='pink')
        image1.image = render
        image1.place(x=400, y=50)
        
    def close_window(): 
        Window.destroy()
    
    def preprocess(self, event=None):
        global rep
        img = cv2.imread(rep[0])
        kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        


        
        self.from_array = Image.fromarray(cv2.resize(sharpened,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)
        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='pink')
        image2.image = render
        image2.place(x=400, y=250)

        gray_img = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        
        entropy_value = shannon_entropy(gray_img)

        # Display entropy value
        entropy_label = Label(self, text="Entropy: {:.2f}".format(entropy_value))
        entropy_label.place(x=600, y=250)  # Adjust coordinates as needed


        # Calculate the squared pixel intensities to get the energy
        energy_value = np.sum(sharpened.astype(np.float32) ** 2)

        # Display the energy value
        energy_label = Label(self, text="Energy: {:.2f}".format(energy_value))
        energy_label.place(x=600, y=300)  # Adjust coordinates as needed

        gray_img = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        skewness_value = skew(gray_img.flatten())  # Compute skewness for the flattened image array

        # Display skewness value
        skewness_label = Label(self, text="Skewness: {:.2f}".format(skewness_value))
        skewness_label.place(x=600, y=350)

        # Calculate and display contrast value
        contrast_value = np.std(gray_img)
        contrast_label = Label(self, text="Contrast: {:.2f}".format(contrast_value))
        contrast_label.place(x=600, y=400)

        

        
    def segmentation(self, event=None):
        
        img_org = cv2.imread(rep[0])
        image=cv2.resize(img_org,(250,250))
       
        
       
  
        # Setting parameter values
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold
  
        # Applying the Canny Edge filter
        edge = cv2.Canny(image, t_lower, t_upper)
        
        #result = cv2.bitwise_and(image, image,mask=mask)
        self.from_array = Image.fromarray(cv2.resize(edge,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)


        # Calculate histogram of pixel values in the segmented image
        hist = cv2.calcHist([edge], [0], None, [256], [0, 256])

        # Normalize histogram
        hist_norm = hist.ravel() / hist.sum()

        # Compute Gini index
        gini_index = 1 - np.sum(np.square(hist_norm))

        # Display Gini index
        gini_label = Label(self, text="Gini Index: {:.4f}".format(gini_index))
        gini_label.place(x=400, y=600)  # Adjust coordinates as needed


        # Calculate GLCM
        distances = [1]  # distance between pixels
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles for GLCM computation
        glcm = graycomatrix(edge, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Calculate homogeneity from GLCM
        homogeneity = graycoprops(glcm, 'homogeneity').mean()

        # Display Homogeneity
        homogeneity_label = Label(self, text="Homogeneity: {:.4f}".format(homogeneity))
        homogeneity_label.place(x=400, y=650)  # Adjust coordinates as needed




        
        

        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='white')
        image4.image = render
        image4.place(x=400, y=450)

        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        img_prewittx = cv2.filter2D(image, -1, kernelx)


        self.from_array = Image.fromarray(cv2.resize(img_prewittx,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)
        

        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='white')
        image5.image = render
        image5.place(x=650, y=450)



        img_sobel = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=5)
        self.from_array = Image.fromarray(cv2.resize(img_sobel,(150,150)))
        render = ImageTk.PhotoImage(self.from_array)
        

        image6=Label(self, image=render,borderwidth=15, highlightthickness=5, height=100, width=100, bg='white')
        image6.image = render
        image6.place(x=850, y=450)

        


        

  
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()





    
    def classification(self, event=None):
        global T,rep
        clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
        from keras.preprocessing import image                  
        
        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)
        from tensorflow.keras.models import load_model
        model = load_model('trained_model_CNN.h5')
        main_img = cv2.imread(rep[0])
        
        test_tensors = paths_to_tensor(rep[0])/255
        pred=model.predict(test_tensors)
        x=np.argmax(pred);
        print('Given image is  = '+clas1[x])
        res='predicted image is '+clas1[x]
        
        T = Text(self, height=5, width=30)
        T.place(x=100, y=500)
        T.insert(END,res)

    

  
                
root = Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()

        
