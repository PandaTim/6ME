import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def drawFigure(loc, img, label):
    plt.subplot(*loc), plt.imshow(img, cmap='gray')
    plt.title(label), plt.xticks([]), plt.yticks([])


if not os.path.exists("results"):
    os.mkdir("results")

font_path = 'C:\\Windows\\Fonts\\나눔고딕.ttf' # 한글을 fig에 담기 위해
font_name = fm.FontProperties(fname=font_path, size=50).get_name()
plt.rc('font', family=font_name)


crop_rate = 0.2
bw_thr = 127
imgs = os.listdir("crawl/1++/")
for img_name in imgs:
    if img_name[-3:] == "png" or img_name[-3:] == "jpg":
        img = cv2.imread("crawl/1++/" + img_name)
        h, w = img.shape[0], img.shape[1]
        (left, right, up, down) = (int(w * (0.5 - crop_rate / 2)), int(w * (0.5 + crop_rate / 2)),
                                   int(h * (0.5 - crop_rate / 2)), int(h * (0.5 + crop_rate / 2)))
        """
        고기(붉은색)와 마블링(흰색)을 구분하기 위해 R채널을 배제하고,
        나머지 채널의 평균값을 구해 thresholding 한다.
        """
        cropped = cv2.resize((img[up:down, left:right, 0].astype("int16")
                              + img[up:down, left:right, 1].astype("int16")) / 2, None,
                             fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        ret, thr = cv2.threshold(cropped, bw_thr, 255, cv2.THRESH_BINARY)
        num = cv2.countNonZero(thr)

        drawFigure((1, 2, 1), cv2.cvtColor(img[up:down, left:right, :], cv2.COLOR_BGR2RGB), 'Original')
        drawFigure((1, 2, 2), thr, 'Thresholded')
        plt.suptitle("마블링 비율: " + str("%.2f" % ((num * 100) / (h * w * pow(crop_rate*2, 2)))) + "%", y=0.2)
        plt.savefig("results/" + img_name)
