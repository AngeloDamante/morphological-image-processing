import os
#import cv2
from PIL import Image


def paste_on_sizes_image(fn, needed_res):
    # Used just for manually align to 10:9
    # call with resize('examples/lena_cameraman.png', (1280, 720), 'examples', 'test.png')
    new_n = os.path.basename(fn)
    img = Image.open(fn)
    img_new = Image.new('L', needed_res)
    img_new.paste(img, (0,0))
    img_new.save(new_n)


def resize(fn, new_size, folder, name):
    img = Image.open(fn)
    img = img.resize(new_size)
    img.save(os.path.join(folder, name))


def save_file_in_all_resolutions(fn, new_folder):
    sizes = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (7680, 4320)]
    if not os.path.isdir(new_folder): os.makedirs(new_folder)
    new_n = os.path.basename(fn)
    for res in sizes:
        print(f"Creating {res[0]}x{res[1]}")
        folder = 'x'.join([str(r) for r in res])
        out_folder = os.path.join(new_folder, folder)
        if not os.path.isdir(out_folder): os.makedirs(out_folder)
        resize(fn, res, out_folder,  new_n[:-4]+'_'+folder+'.png')


if __name__ == '__main__':
    folder_images = 'examples'
    for file in os.listdir(folder_images):
        print(f"Processing image {file}")
        save_file_in_all_resolutions(os.path.join(folder_images, file), 'images')