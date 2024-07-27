import cv2

# Make sure the path to the HAAR cascade file is correct
haar_cascade_path = 'haarcascade_frontalface_default.xml'

# Load the cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if the cascade file is loaded correctly
if face_cascade.empty():
    print("Error: Could not load HAAR cascade file. Check the path.")
else:
    # Read the input image
    img_path = 'gp.jpg'  # Update this to the correct path of your image
    img = cv2.imread(img_path)

    # Check if the image is loaded correctly
    if img is None:
        print(f"Error: Could not load image at {img_path}. Check the path.")
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the output
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
