#/bin/bash
echo "Downloading images"
wget https://github.com/AngeloDamante/morphological-image-processing-in-parallel/releases/download/0.1/images_example_grayscale_10.tar.gz
tar -zxvf images_example_grayscale_10.tar.gz
echo "Expanding images"
python3 expand_images.py
echo "Removing temporany datas"
rm examples -r
rm images_example_grayscale_10.tar.gz

echo "Downloading CImage lib"
wget https://github.com/AngeloDamante/morphological-image-processing-in-parallel/releases/download/0.2/CImg.tar.gz
tar -zxvf CImg.tar.gz
mv CImg.h parallel_CUDA/tool/
rm CImg.tar.gz

echo "Done"
