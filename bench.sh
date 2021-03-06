#!/bin/bash
BENCHMARKFOLDER="benchmarks"

if [[ "$0" == "./bench.sh" ]]; then

WARPSIZE=32
MAXBLOCKSIZE=1024
filterfunc=$1
if [ "$#" -eq 4 ]
then
howmanytimes=$2
filetosave=$3
winWidth=$4
elif [ "$#" -eq 3 ]
then
howmanytimes=$2
filetosave=$3
winWidth=3
else
howmanytimes=1
filetosave=$filterfunc
winWidth=3
fi
logfile="$BENCHMARKFOLDER/$filetosave.log"
imginpath=img_in/*

bench1d() {
imgpath=$1

imgbasename=$(basename $imgpath)

imgname="${imgbasename%%.*}"
imgext="${imgbasename#*.}"

echo $imgbasename >> $logfile

for ((blocksize = $WARPSIZE ; blocksize <= $MAXBLOCKSIZE ; blocksize+=$WARPSIZE))
do
printf "$blocksize " >> $logfile
done
printf "\n" >> $logfile

for ((blocksize = $WARPSIZE ; blocksize <= $MAXBLOCKSIZE ; blocksize+=$WARPSIZE))
do
newimg="${imgname}_${filterfunc}_${blocksize}.${imgext}"
val=$(./build/main ./$imgpath ./img_out/$newimg $filterfunc $blocksize $winWidth)
printf "$val " >> $logfile
done
printf "\n" >> $logfile
}

rm -f $logfile

imgfiles=$(ls $imginpath | wc -l)
echo $howmanytimes >> $logfile
for ((i = 0 ; i < $howmanytimes ; i++))
do
echo $imgfiles >> $logfile
for imgpath in $imginpath
do
set -x
bench1d $imgpath
set +x
done
done

fi