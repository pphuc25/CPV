import cv2

# ip = 'rtsp://admin:admin@192.168.2.2::/cam/realmonitor?channel=1&subtype=1'
ip = "rtsp://admin:ZSXNWK@192.168.225.155:554/H.264"

cap = cv2.VideoCapture(ip)

# Define the cropping coordinates (left, top, right, bottom)
crop_coords = (1080//2 - 360, 1920//2 - 480, 1080//2 + 360, 1920//2 + 480)

# while True:
#
#     #Capture the frame
#     ret, frame = cap.read()
#     width, height = frame.shape[:2]
#
#     frame = frame[crop_coords[0] : crop_coords[2], crop_coords[1] : crop_coords[3]]
#
#     cv2.imshow('IP', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()