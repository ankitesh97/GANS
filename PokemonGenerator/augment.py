
from PIL import Image
import os

MAIN_DIR = 'data/pokemonResized/'
FLIPPED = 'data/flipped'
ROTATED = 'data/rotated'


def flip_image(image):
    new_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    return new_img


def rotate_image(image, degree):
    new_img = image.convert('RGBA') # convert to RGBA to get transparent background
    new_img = new_img.rotate(degree)
    background = Image.new("RGB", new_img.size, (255, 255, 255))
    background.paste(new_img, mask=new_img.split()[3]) # 3 is the alpha channel
    return background


def flip_all_images():
    for each in os.listdir(MAIN_DIR):
        img = Image.open(MAIN_DIR + '/' + each, 'r')
        img.load()
        flip_img = flip_image(img)

        flip_img.save(FLIPPED+'/flipped_'+each, 'JPEG', quality=100)


def rotate_all_images():
    for each in os.listdir(FLIPPED):
        img = Image.open(FLIPPED + '/' + each, 'r')
        img.load()
        img.save(ROTATED+'/'+each)
        rotate3deg = rotate_image(img, 8)
        rotate3deg.save(ROTATED+'/rotated_degree8_'+each, 'JPEG', quality=100)

        rotate3deg = rotate_image(img, 10)
        rotate3deg.save(ROTATED+'/rotated_degree10_'+each, 'JPEG', quality=100)

        rotate3deg = rotate_image(img, -8)
        rotate3deg.save(ROTATED+'/rotated_degree-8_'+each, 'JPEG', quality=100)

        rotate3deg = rotate_image(img, -10)
        rotate3deg.save(ROTATED+'/rotated_degree-10_'+each, 'JPEG', quality=100)


# flip_all_images()
rotate_all_images()
