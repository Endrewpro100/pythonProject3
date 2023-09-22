import cv2
from PIL import Image

image_cat="cat.jpeg"
cat=Image.open(image_cat)
image_glasses="glasses.png"
glasses=Image.open(image_glasses)
image_rot="rot.png"
rot=Image.open(image_rot)
image_cep="cep.png"
cep=Image.open(image_cep)
cat=cat.convert("RGBA")
glasses=glasses.convert("RGBA")
rot=rot.convert("RGBA")
cep=cep.convert("RGBA")
cat_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")
img=cv2.imread(image_cat)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cat_face=cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in cat_face:
    glasses = glasses.resize((w, int(h / 3)))
    cat.paste(glasses, (x, int(y + h / 4)), glasses)
    rot=rot.resize((w, int(h / 3)))
    cat.paste(rot, (x, int(y + h / 1.6)), rot)
    cep=cep.resize((w, int(h / 2)))
    cat.paste(cep, (x-4, int(y + h / 1.2)), cep)
cat.save("kit.png")
kit=cv2.imread("kit.png")
cv2.imshow("cat", kit)
cv2.waitKey(0)
cv2.destroyAllWindows()
