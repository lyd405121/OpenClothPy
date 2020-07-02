#gif.py 
import sys
import datetime
import imageio

VALID_EXTENSIONS = ('png', 'jpg')


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'cloth.gif'
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    filenames = []
    for i in range(0,300):
        filenames.append(str(i)+ ".png")

    create_gif(filenames, 0.03)