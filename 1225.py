import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
from PIL import Image, ImageDraw

image_template = mpimg.imread('template1.bmp')
image_template2 = mpimg.imread('template2.bmp')
image_org = mpimg.imread('test image 1.bmp')
org = Image.open('test image 1.bmp')

# image_template = mpimg.imread('101_temp.jpg')
# image_template2 = mpimg.imread('101_temp.jpg')
# image_org = mpimg.imread('target1.jpg')

# org =Image.open('101_temp.jpg')


image_template = np.array(image_template)
print(np.shape(image_template))
image_template2 = np.array(image_template2)
image_org = np.array(image_org)
print(image_template.shape)

# image_org = mpimg.imread('C:/Users/user/Desktop/課程/大四上/機器視覺/Final_project/template image.bmp')
# print(image_org)
image_org.flags.writeable = True
image_template.flags.writeable = True

def filter(image,filtersize,filter):
    k = 3
    m,n = image.shape[0], image.shape[1]
    # print(m,n,k)
    diameter = int(filtersize/2)
    for row1 in range(diameter,m-diameter,1):
        for column1 in range(diameter,n-diameter,1):
            sum = [0,0,0]
            i = 0
            for row2 in range(row1-diameter,row1+diameter+1,1):
                j=0
                for column2 in range(column1-diameter,column1+diameter+1,1):
                    sum = sum + np.dot(filter[i,j],image[row2,column2])
                    j = j + 1
                i=i+1
            image[row1,column1] = sum
    return image

kernel = np.array([
	        [0.045,0.122,0.045],
	        [0.122,0.332,0.122],
	        [0.045,0.122,0.045]])


# image_org = filter(image_org,3,kernel)
image_org = filter(image_org,3,kernel)

image_template = filter(image_template,3,kernel)

# plt.figure(figsize=(30,10))
# plt.imshow(image_org), plt.axis('off')
#
# plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    # return np.dot(rgb[..., :3], [1, 0, 0])



if len(image_org.shape) == 3:
    gray = rgb2gray(image_org)
    template1 = rgb2gray(image_template)
    template2 = rgb2gray(image_template2)

# plt.figure(figsize=(30,10))
# plt.imshow(gray,cmap='gray')
# plt.axis('off')
# plt.show()


m,n = np.shape(gray)
#print(m,n)


t,e = np.shape(template1)
#print(t,e)
t2,e2 = np.shape(template2)

# print(m,n,t,e,t2,e2)

def candicate_position(diatance,window_center):
    position = np.array([[window_center[0]-diatance,window_center[1]-diatance],[window_center[0]-diatance,window_center[1]],[window_center[0]-diatance,window_center[1]+diatance],
                          [window_center[0],window_center[1]-diatance],[window_center[0],window_center[1]],[window_center[0],window_center[1]+diatance],
                          [window_center[0]+diatance,window_center[1]-diatance],[window_center[0]-diatance,window_center[1]],[window_center[0]-diatance,window_center[1]+diatance]])

    return position

def position_image(template,point_i,point_j):
    i,j = np.shape(template)
    if i % 2 == 0:
        size_x = [point_i - i/2, point_i + i / 2]
    else:
        size_x = [point_i - int(i / 2), point_i + 1+ int(i / 2)]
    if j % 2 == 0:
        size_y = [point_j - j / 2, point_j + j / 2]
    else:
        size_y = [point_j - int(j / 2), point_j + 1 + int(j / 2)]
    return size_x,size_y


def SSD(template,image,position):
    i,j = np.shape(template)
    # print(position)
    point_i = position[0]
    point_j = position[1]
    size_x, size_y = position_image(template,point_i,point_j)
    # print(size_x,size_y)
    x=0
    y=0
    ssd =0
    for row1 in range(int(size_x[0]),int(size_x[1])) :
        if i>x:
            for column1 in range(int(size_y[0]),int(size_y[1])) :
                if j>y :
                    ssd = ssd + np.square(image[row1,column1]-template[x,y])
                y = y + 1
        x = x + 1
    return ssd

def three_step_search_SSD(window_center,template):
    # print(window_center)
    # First Step
    position1 = candicate_position(4,window_center)
    # print(position)
    ssd = [0 for i in range(9)]
    for i in range(9):
        ssd[i] = SSD(template,gray,position1[i])
    # print(ssd)
    position_center1 = position1[np.argmin(ssd)]
    # print(position_center1)

    # Second Step
    position2 = candicate_position(2,position_center1)
    # print(position)
    ssd = [0 for i in range(9)]
    for i in range(9):
        ssd[i] = SSD(template,gray,position2[i])
    # print(ssd)
    position_center2 = position2[np.argmin(ssd)]
    # print(position_center2)

    # Third Step
    position3 = candicate_position(1,position_center2)
    # print(position)
    ssd = [0 for i in range(9)]
    for i in range(9):
        ssd[i] = SSD(template,gray,position3[i])
    # print(ssd)
    position_center3 = position3[np.argmin(ssd)]
    # print(position_center3)
    min_ssd = min(ssd)
    min_position = position_center3
    return min_position,min_ssd

def graylevel(image):
    histogram = [0 for i in range(0,256)]
    i,j = np.shape(image)
    for row in range(i):
        for column in range(j):
            id = int(image[row,column])
            histogram[id] = histogram[id] + 1
    return histogram
# print(histogram_template)

def PDF(template,image,position):
    i,j = np.shape(template)
    # print(position)
    point_i = position[0]
    point_j = position[1]
    size_x, size_y = position_image(template,point_i,point_j)
    # print(size_x,size_y)
    histogram = [0 for i in range(0,256)]
    for row1 in range(int(size_x[0]),int(size_x[1])) :
        for column1 in range(int(size_y[0]),int(size_y[1])) :
                id = int(image[row1, column1])
                histogram[id] = histogram[id] + 1
    return histogram

def three_step_search_PDF(window_center,template):
    # First Step
    position1 = candicate_position(4,window_center)
    # print(position)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template,gray,position1[i])
        histogram_template = graylevel(template)
        for m in range(256):
            Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i]+(histogram_gray[m]*histogram_template[m])**0.5
    position_center1 = position1[np.argmax(Bhattacharyya_coefficient)]
    # print(position_center1)

    # Second Step
    position2 = candicate_position(2,position_center1)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template, gray, position2[i])
        histogram_template = graylevel(template)
        for m in range(256):
            Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + (
                        histogram_gray[m] * histogram_template[m]) ** 0.5
    position_center2 = position2[np.argmax(Bhattacharyya_coefficient)]

    # Third Step
    position3 = candicate_position(1,position_center2)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template, gray, position3[i])
        histogram_template = graylevel(template)
        for m in range(256):
            Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + (
                    histogram_gray[m] * histogram_template[m]) ** 0.5
    position_center3 = position3[np.argmax(Bhattacharyya_coefficient)]
    # print(position_center3)
    max_coefficient = max(Bhattacharyya_coefficient)
    max_position = position_center3
    return max_coefficient,max_position


def window_search(image,template,is_PDF=False,is_SSD=False):
    m1,n1 = np.shape(template)
    m2,n2 = np.shape(image)
    min_ssd = 0
    min_position = 0
    max_coefficient = 0
    max_position = 0
    for row in range(int(m1/2),int(m2-m1/2)-10,15):
        if row < int(m2-m1/2)-10 :
            for column in range(int(n1/2),int(n2-n1/2)-10,15):
                if column < int(n2-n1/2)-10:
                    if is_SSD == True:
                        # print(row,column)
                        posion,ssd = three_step_search_SSD(window_center=[row,column],template=template)
                        # print(ssd)
                        if ssd < min_ssd or min_ssd == 0 :
                            min_ssd = ssd
                            min_position = posion

                    if is_PDF == True:
                        coefficient, position = three_step_search_PDF(window_center=[row,column],template=template)
                        if coefficient > max_coefficient or max_coefficient == 0 :
                            max_coefficient = coefficient
                            max_position = position
    return min_ssd,min_position,max_coefficient,max_position

# SSD + TSS
ssd,position,void1,void2 = window_search(gray,template1,is_SSD=True)
ssd2,position2,void1,void2 = window_search(gray,template2,is_SSD=True)

# print(ssd,position)
rectangle_x, rectangle_y = position_image(template1,position[0],position[1])
rectangle_x12, rectangle_y12 = position_image(template2,position[0],position[1])

# print(rectangle_x,rectangle_y)

# org =Image.open('template image.bmp')

draw = ImageDraw.Draw(org)
if ssd < ssd2:
    draw.line([(int(rectangle_y[0]),int(rectangle_x[0])),
               (int(rectangle_y[0]),int(rectangle_x[1])),
               (int(rectangle_y[1]),int(rectangle_x[1])),
               (int(rectangle_y[1]),int(rectangle_x[0])),
               (int(rectangle_y[0]),int(rectangle_x[0]))
               ]
               , width=2, fill=(255,0,0))
else:
    draw.line([(int(rectangle_y12[0]),int(rectangle_x12[0])),
               (int(rectangle_y12[0]),int(rectangle_x12[1])),
               (int(rectangle_y12[1]),int(rectangle_x12[1])),
               (int(rectangle_y12[1]),int(rectangle_x12[0])),
               (int(rectangle_y12[0]),int(rectangle_x12[0]))
               ]
           , width=2, fill=(255,255,0))
# draw.line([(int(rectangle_y[0]),int(rectangle_x[0])),
#                (int(rectangle_y[0]),int(rectangle_x[1])),
#                (int(rectangle_y[1]),int(rectangle_x[1])),
#                (int(rectangle_y[1]),int(rectangle_x[0])),
#                (int(rectangle_y[0]),int(rectangle_x[0]))
#                ]
#                , width=2, fill=(255,0,0))
#
# draw.line([(int(rectangle_y12[0]),int(rectangle_x12[0])),
#                (int(rectangle_y12[0]),int(rectangle_x12[1])),
#                (int(rectangle_y12[1]),int(rectangle_x12[1])),
#                (int(rectangle_y12[1]),int(rectangle_x12[0])),
#                (int(rectangle_y12[0]),int(rectangle_x12[0]))
#                ]
#            , width=2, fill=(255,255,0))
org.show()
del draw


# PDF + TSS
void1,void2,max_coefficient,max_position = window_search(gray,template1,is_PDF=True)
void1,void2,max_coefficient2,max_position2 = window_search(gray,template2,is_PDF=True)

# print(ssd,position)
rectangle_x2, rectangle_y2 = position_image(template1,max_position[0],max_position[1])
rectangle_x22, rectangle_y22 = position_image(template2,max_position[0],max_position[1])

# print(rectangle_x,rectangle_y)
# org =Image.open('test image 2.bmp')
# org =Image.open('C:/Users/user/Desktop/課程/大四上/機器視覺/Final_project/template image.bmp')

draw = ImageDraw.Draw(org)
if max_coefficient > max_coefficient2 :
    draw.line([(int(rectangle_y2[0]),int(rectangle_x2[0])),
               (int(rectangle_y2[0]),int(rectangle_x2[1])),
               (int(rectangle_y2[1]),int(rectangle_x2[1])),
               (int(rectangle_y2[1]),int(rectangle_x2[0])),
               (int(rectangle_y2[0]),int(rectangle_x2[0]))
               ]
               , width=2, fill=(255,0,0))
else:
    draw.line([(int(rectangle_y22[0]),int(rectangle_x22[0])),
               (int(rectangle_y22[0]),int(rectangle_x22[1])),
               (int(rectangle_y22[1]),int(rectangle_x22[1])),
               (int(rectangle_y22[1]),int(rectangle_x22[0])),
               (int(rectangle_y22[0]),int(rectangle_x22[0]))
               ]
               , width=2, fill=(255,255,0))

# draw.line([(int(rectangle_y2[0]),int(rectangle_x2[0])),
#                (int(rectangle_y2[0]),int(rectangle_x2[1])),
#                (int(rectangle_y2[1]),int(rectangle_x2[1])),
#                (int(rectangle_y2[1]),int(rectangle_x2[0])),
#                (int(rectangle_y2[0]),int(rectangle_x2[0]))
#                ]
#                , width=2, fill=(255,0,0))
#
# draw.line([(int(rectangle_y22[0]),int(rectangle_x22[0])),
#                (int(rectangle_y22[0]),int(rectangle_x22[1])),
#                (int(rectangle_y22[1]),int(rectangle_x22[1])),
#                (int(rectangle_y22[1]),int(rectangle_x22[0])),
#                (int(rectangle_y22[0]),int(rectangle_x22[0]))
#                ]
#                , width=2, fill=(255,255,0))
org.show()
del draw



# plt.figure(figsize=(30,10))
# plt.imshow(gray), plt.axis('off')
#
#
# plt.show()


# gray_level = [i for i in range(0,256)]
# histogram = [0 for i in range(0,256)]
# for row in range(t2):
#     for column in range(e2):
#         id = int(template2[row,column])
#         histogram[id] = histogram[id] + 1
#
#
#
#
#
# plt.bar(gray_level,histogram)
# plt.title("histogram")
# plt.show()

