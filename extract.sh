#!/bin/sh

input_folder='./stackexchange'
output_folder='./extracted'
data_folder='./data'

echo "Will delete $output_folder and $data_folder folder in the project directory (if it exists) and create a new folder with extracted stackexchange data"
read -p "Continue[y/n]? " reply
case "$reply" 
in y)
    if [ -d $output_folder ] 
    then
        rm -r $output_folder
    fi

    if [ -d $data ] 
    then
        rm -r $data_folder
    fi
    mkdir -p $output_folder
    mkdir -p $data_folder

    for f in $input_folder/*; do
        name="$(echo $f | cut -d'/' -f 3)"
        extension=${name##*.}
        if [ $extension = "7z" ]; then
            folder="${name%.*}"
            mkdir -p $output_folder/$folder
            mkdir -p $data_folder/$folder
            7za x -o$output_folder/$folder $f
        fi
    done
    ;;
*) 
    echo "Exiting..." 
    exit;;
esac