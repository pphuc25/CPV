import cv2

def resize(img, percent):
    #resize to how many percent of the original image
    scale_percent = percent
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height) 
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def image_stitching(img_1, img_2, percent=100):

    imgs = [img_1, img_2]
    
    stitchy=cv2.Stitcher.create()
    (dummy,output)=stitchy.stitch(imgs)
    
    if dummy != cv2.STITCHER_OK:
        print("Stitching ain't successful")
    else: 
        print('Your Panorama is ready!!!')
    resized = resize(output, percent) #resize image to be able to see a whole image
    return resized

def on_button_click(event, x, y, flags, param):
    global stitching_image
    global button_hover
    global button_text

    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x and x <= 10 + button_size[0] and 10 <= y and y <= 10 + button_size[1]:
            if button_text == 'STITCHING':
                print("STITCHING Image")
                button_text = 'RAW'
                stitching_image = image_stitching(img_1, img_2)
            else:
                print('RAW')
                button_text = 'STITCHING'
                stitching_image = cv2.hconcat([img_1, img_2])

url1 = 'images\P1010517.JPG'
url2 = 'images\P1010520.JPG'

img_1 = cv2.imread(url1)
img_2 = cv2.imread(url2)

# Create the window
window_name = 'STITCHING Image'
cv2.namedWindow(window_name)

# Combine the images horizontally
stitching_image = cv2.hconcat([img_1, img_2])

# Get the size of the combined image
height, width, _ = stitching_image.shape

# Define button properties
button_text = "STITCHING"
button_font = cv2.FONT_ITALIC
button_font_scale = 0.6
button_thickness = 1
button_padding = 10
button_color = (255, 0, 0)
button_hover_color = (0, 255, 0)

# Get button size and position
button_size, _ = cv2.getTextSize(button_text, button_font, button_font_scale, button_thickness)
button_rect = ((0, 0), (button_size[0] + button_padding*2, button_size[1] + button_padding*2))

# Initialize button hover state
button_hover = False

# Show the image with the button
cv2.setMouseCallback(window_name, on_button_click)

# Wait for a button click
while True:
    #Display the combined image
    cv2.imshow(window_name, stitching_image)

    # Draw button
    if button_hover:
        button_text_color = button_hover_color
    else:
        button_text_color = button_color
    cv2.rectangle(stitching_image, button_rect[0], button_rect[1], button_text_color, -1)
    cv2.putText(stitching_image, button_text,
                (button_rect[0][0] + button_padding, button_rect[0][1] + button_size[1] + button_padding), button_font,
                button_font_scale, (255, 255, 255), button_thickness, cv2.LINE_AA)

    # Check for key events
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Close the window
cv2.destroyAllWindows()