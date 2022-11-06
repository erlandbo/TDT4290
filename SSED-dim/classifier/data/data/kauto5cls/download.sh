#!/usr/bin/env bash


#Move over to directory of this script
cd "$(dirname "$0")"

echo ""
echo "#####################################"
echo "#         DOWNLOADING DATA          #"
echo "#####################################"
echo ""

fileId=1mPApAnXBt7SFE8jI53rQnzCaFQd5Dd4U
fileName=dataset.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}


echo ""
echo "#####################################"
echo "#         UNZIPPING DATASETS        #"
echo "#####################################"
echo ""

unzip dataset.zip


echo ""
echo "#####################################"
echo "#       CLEANING UP ZIP-FILE        #"
echo "#####################################"
echo ""

rm dataset.zip
