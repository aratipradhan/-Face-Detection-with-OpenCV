import cv2

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the image
image_path = r"D:\Python tasks_Ds\Sentiment Detection\Testing\group_puri.png"
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path.")
    exit()


# Get screen dimensions (assuming 1920x1080 screen resolution)
screen_width = 1280  # Adjust based on your screen's width
screen_height = 720  # Adjust based on your screen's height
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Check if any faces are detected
if len(faces) == 0:
    print("No faces found")
else:
    # Draw rectangles and add text above each detected face
    for (x, y, w, h) in faces:
        # Draw the rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)
        
        # Write 'Face' above the rectangle
        text_position = (x, y - 10)  # Position above the rectangle
        cv2.putText(image, 'Face', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

# Resize the image to fit the screen (optional)
image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)

# Display the result
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
