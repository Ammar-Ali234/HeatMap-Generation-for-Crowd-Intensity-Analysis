import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("/content/test2.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("heatmap_output2_r1.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))



# Initialize heatmap object
heatmap = solutions.Heatmap(
    show=True,  # display the output
    model="yolo11n.pt",  # path to the YOLO11 model file
    colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
    #region=region_points,  # object counting with heatmaps, you can pass region_points
    classes=[2,3,5,7],  # generate heatmap for specific classes i.e person and car.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = heatmap(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows