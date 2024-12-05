def segmentation(self, event=None):
    global rep
    img = cv2.imread(rep[0])
    
    # Creating kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Using cv2.erode() method 
    image = cv2.erode(img, kernel)
    
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold
  
    # Applying the Canny Edge filter
    edge = cv2.Canny(img, t_lower, t_upper)
    
    # Find contours in the segmented image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw the contours on
    marked_img = img.copy()
    
    # Draw contours around the segmented area
    cv2.drawContours(marked_img, contours, -1, (0, 255, 0), 2)  # Green color
    
    self.from_array = Image.fromarray(cv2.resize(edge, (200, 200)))
    render = ImageTk.PhotoImage(self.from_array)
    
    image3 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
    image3.image = render
    image3.place(x=700, y=450)

    self.from_array = Image.fromarray(cv2.resize(marked_img, (200, 200)))
    render = ImageTk.PhotoImage(self.from_array)
    
    image4 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
    image4.image = render
    image4.place(x=950, y=450)
