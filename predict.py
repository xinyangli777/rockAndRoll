from keras.layers import Input
from mask_rcnn import MASK_RCNN 
from PIL import Image

mask_rcnn = MASK_RCNN()

while True:
    img = input('输入img文件夹中的测试图片名:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        image = mask_rcnn.detect_image(image)

        image.show()
mask_rcnn.close_session()