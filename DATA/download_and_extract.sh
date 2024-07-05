#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <password>"
    exit 1
fi

PASSWORD=$1

# Create the target directories
mkdir -p ./DIVO_unprocessed/images/test
mkdir -p ./DIVO_unprocessed/images/train

# URLs to download
urls=(
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Circle_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Circle_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Circle_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Floor_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Floor_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Floor_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Gate1_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Gate1_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Gate1_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Gate2_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Gate2_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Gate2_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Ground_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Ground_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Ground_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Moving_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Moving_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Moving_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Park_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Park_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Park_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Shop_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Shop_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Shop_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Side_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Side_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Side_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Square_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Square_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/train/Square_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Circle_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Circle_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Circle_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Floor_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Floor_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Floor_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Gate1_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Gate1_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Gate1_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Gate2_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Gate2_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Gate2_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Ground_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Ground_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Ground_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Moving_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Moving_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Moving_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Park_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Park_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Park_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Shop_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Shop_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Shop_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Side_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Side_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Side_View3.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Square_View1.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Square_View2.zip"
    "https://huggingface.co/datasets/syhao777/DIVOTrack/resolve/main/datasets/images/test/Square_View3.zip"
)


# Download the files
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    if [[ "$url" == *"/test/"* ]]; then
        wget -O ./DIVO_unprocessed/images/test/"$filename" "$url"
    elif [[ "$url" == *"/train/"* ]]; then
        wget -O ./DIVO_unprocessed/images/train/"$filename" "$url"
    fi
done

# Unzip the files with the password using unar

for file in ./DIVO_unprocessed/images/test/*.zip; do
    unar -p "$PASSWORD" -o ./DIVO_unprocessed/images/test "$file"
    rm "$file"
done

for file in ./DIVO_unprocessed/images/train/*.zip; do
    unar -p "$PASSWORD" -o ./DIVO_unprocessed/images/train "$file"
    rm "$file"
done

# Clone the testset labels folder from DIVOTrack GitHub
git clone --depth 1 https://github.com/shengyuhao/DIVOTrack.git
cd DIVOTrack
git sparse-checkout init --cone
git sparse-checkout set MOTChallengeEvalKit_cv_test/self
cp -r MOTChallengeEvalKit_cv_test/self ../DIVO_unprocessed/
cd ..
rm -rf DIVOTrack


echo "Download, extraction, and cleanup complete."
