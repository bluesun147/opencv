import cv2

# https://www.youtube.com/watch?v=VtVPLJODqKw&list=PLwfJJiO20qkDue05S5MNhgYNnClMVlagN&index=4
# 이진화 (black and white)

def nothing(x):
    pass

cv2.namedWindow('Binary')
cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'Binary', 97)

img_color = cv2.imread('red_ball.jpg', cv2.IMREAD_COLOR)

cv2.imshow("Color", img_color)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', img_gray)
cv2.waitKey(0)

while(True):
    low = cv2.getTrackbarPos('threshold', 'Binary')
    ret,img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Binary', img_binary)

    img_result = cv2.bitwise_and(img_color, img_color, mask = img_binary)
    cv2.imshow('Result', img_result)
    if cv2.waitKey(1)&0xFF == 27:
        break

cv2.destroyAllWindows()