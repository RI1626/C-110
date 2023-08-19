# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model("keras_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    check, frame = vid.read()
    
    #resizeing Image
    img=cv2.resize(frame,(224,224))

    #convert image iNTO nUNPY ARRAY
    testimage=np.array(img,dtype=np.float32)
    testimage=np.expand_dims(testimage,axis=0)

    #normallising the image
    normalizedimage=testimage/255.0
    
    #predict result
    prediction=model.predict(normalizedimage)

    print("prediction",prediction)


    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()