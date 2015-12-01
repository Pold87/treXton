#! /bin/bash
echo "###############"
echo "This is treXton"
echo "###############"
echo ""


##################
# Settings
##################

# Basedir of the images (this is basically the only important thing)
basedir=~/Documents/Internship/treXton/imgs/

# Create an orthomap using hugins
create_orthomap=false

# Rename files in directory from 0 to n.png
rename_files=false

# Do ground truth labeling using SIFT
sift_gtl=false

# Run draug for folder (augment image views)
draug=false

# Path to map (for matching SIFT)
mymap=~/Documents/Internship/draug/img/bestnewmat.png

# Path to ground truth creator
gtl=~/Documents/Internship/treXton/ground_truth_creator.py

# Path to treXton
treXton=~/Documents/Internship/treXton/treXton.py

# Use treXton (perform regression) 
use_treXton=true

##################
# Renaming process
##################

if $rename_files ; then
echo "Start: Renaming files"

# To avoid renaming files to already existing files create temporary
# file names first
cd $basedir
a=0
for i in *.png; do
  new=$(printf "${basedir}%d_tmp.png" "$a")
  mv -- "$i" "$new"
  let a=a+1
done

# Actually rename files
a=0
for i in *.png; do
  new=$(printf "${basedir}%d.png" "$a")
  echo "$new"
  mv -- "$i" "$new"
  let a=a+1
done

echo "Finished: Renaming files"
fi

##################
# Stitching
##################


if $create_orthomap ; then
echo "Start: Creating orthomap"

# Create folder for orthomap-related stuff
mkdir orthomap

# Move files to new folder
cp *{00..99..10}.png orthomap

cd orthomap
pto_gen -o project.pto *.png
cpfind -o project.pto --multirow --celeste project.pto
cpclean -o project.pto project.pto
autooptimiser -a -m -l -s -o project.pto project.pto
pano_modify --canvas=AUTO --crop=AUTO -o project.pto project.pto
hugin_executor --prefix=prefix project.pto
nona -m TIFF_m -o project project.pto
enblend -o project.tif *.tif

echo "Finished: Creating orthomap"

fi

###########################
# SIFT Ground truth labeler
###########################

if $sift_gtl ; then

echo "Starting: Matching using SIFT ground truth labeler"

$gtl --mymap $mymap --basedir $basedir

echo "Finished: Matching using SIFT ground truth labeler"

mv sift_targets.csv ..

fi

###########################
# Draug for folder
###########################

if $draug ; then

echo "Starting: Running draug for folder"

draug_for_folder 5 $basedir

echo "Finished: Running draug for folder"

fi


##############################
# Perform regression (treXton)
##############################

if $use_treXton ; then

echo "Starting: Performing regression (treXton)"

$treXton

echo "Finished: Performing regression (treXton)"

fi
