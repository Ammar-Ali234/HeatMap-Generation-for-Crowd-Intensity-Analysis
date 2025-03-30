import cv2
from ultralytics import solutions

# Open video file
cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define video writer (MP4 format)
video_writer = cv2.VideoWriter("heatmap_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize heatmap generator
heatmap = solutions.Heatmap(
    show=True,  # Show live visualization
    model="yolo_nas_s.pt",  # Ensure correct YOLO-NAS model file
    colormap=cv2.COLORMAP_JET,  # Heatmap color style
    classes=[2, 3, 5, 7],  # Vehicle classes (Car, Motorcycle, Bus, Truck)
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video processing complete.")
        break

    # Apply heatmap processing
    results = heatmap(im0)

    # Display results live
    cv2.imshow("Heatmap", results.plot_im)

    # Save processed frame
    video_writer.write(results.plot_im)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
video_writer.release()
cv2.destroyAllWindows()
