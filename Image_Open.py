from PIL import Image

# Load Image
im = Image.open('images\ISIC_0026590.jpg')

# Visualize Image
im.show()

# Image Dimensions
print(im.size) 

# Rotate Image
im_rotate = im.rotate(45)
im_rotate.show()