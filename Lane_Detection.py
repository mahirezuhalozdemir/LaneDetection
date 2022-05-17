import numpy as np
import cv2
import matplotlib.pyplot as plt

#gray scale(1)
def Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

#maskeleme işlemi(2)
def Masking(img):
    #white masking
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    lower_thresholdw=np.uint8([243, 249, 243)])
    upper_thresholdw=np.uint8([255,255,255])
    white_mask = cv2.inRange(img_rgb,lower_thresholdw,upper_thresholdw)
    lower_threshold = np.uint8([175, 175, 0])
    upper_threshold = np.uint8([255, 255, 255])
    #sarı maskeleme işlemini orjinal haliyle yapmalı
    yellow_mask = cv2.inRange(img_rgb, lower_threshold, upper_threshold)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    #beyaz ve sarı maskeli halini ekledim
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#gaussian blur(3)
def Gauss(img):
    #ksize=5
    gb = cv2.GaussianBlur(img, (5, 5), 0.8)
    return gb

#canny(4)
def Canny(img):
    canny = cv2.Canny(img, 50, 150)
    #resim,minvalue,maxvalue
    return canny

def LaneDetection(img):
    #resmin uzunluğu ve genişliği
    height = img.shape[0]
    width = img.shape[1]
    #ilgilenilen bölge

    region_ratios= [
        (0,1),
        (0.45,0.60),
        (0.55,0.60),
        (1,1)
    ]
    region= [(int(ratio[0]* width),int(ratio[1]* height)) for ratio in region_ratios]
    vertices=np.array([[(width/2,height/10),(width,height/10),(width,height),(0,height)]],np.int32)
    #vertices = np.array([[(150, 525), (440, 320), (520, 330), (920, 525)]], np.int32)
    cropped_image = region_of_interest(img,vertices)
    #çizgileri HoughLinesp metodu çizer
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=15, minLineLength=40, maxLineGap=20)
    #parametreler:
    #resim, çizgiler,rho,theta,threshold,min hat uzunluğu,çizgiyi bağlama mesafesi
    #lines_img = draw_lines(img, lines)
    return lines

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = (255,255,255)
    cv2.fillPoly(mask, vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#çizgileri videoya ekleme
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (250,170, 0), 2)
            #resim,baslama noktasi,bitis noktasi,renk,kalınlık
            #turuncu renkte çizgi
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    #iki resmi birleştirir
    return img


def video():
    # video okuma fonksiyonu
    video = cv2.VideoCapture("Road.mp4")
    while (video.isOpened()):
        # video okuma işlemi devam ederken
        ret, frame = video.read()
        gray_frame = Gray(frame)
        masking_frame = Masking(gray_frame)
        gauss_frame = Gauss(masking_frame)
        canny_frame = Canny(gauss_frame)
        lines = LaneDetection(canny_frame)
        result = draw_lines(frame, lines)
        # frame çiz
        cv2.imshow('Lanes Detection', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def photo():
    frame1 = cv2.imread("road.jpg")
    frame=cv2.resize(frame1,(700,500),interpolation = cv2.INTER_AREA )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = Gray(frame)
    masking_frame = Masking(gray_frame)
    gauss_frame = Gauss(masking_frame)
    canny_frame = Canny(gauss_frame)
    lines = LaneDetection(canny_frame)
    result = draw_lines(frame, lines)
    # frame çiz
    cv2.imshow('Lanes Detection', result)
    cv2.waitKey(6000)


if __name__ == "__main__":
    photo()