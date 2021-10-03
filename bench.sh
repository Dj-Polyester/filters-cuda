
WARPSIZE=32
filterfunc=$1

bench1d() {
imgpath=$1

imgbasename=$(basename $imgpath)

imgname="${imgbasename%%.*}"
imgext="${imgbasename#*.}"

logfile="logs/$filterfunc.log"

echo $imgbasename >> $logfile

blocksize=$WARPSIZE
while [ $blocksize -lt 544 ]
do
printf "$blocksize " >> $logfile
((blocksize+=$WARPSIZE))
done
printf "\n" >> $logfile

blocksize=$WARPSIZE
while [ $blocksize -lt 544 ]
do
newimg="${imgname}_${filterfunc}_${blocksize}.${imgext}"
val=$(./build/main ./$imgpath ./img_out/$newimg $filterfunc $blocksize)
printf "$val " >> $logfile
((blocksize+=$WARPSIZE))
done
printf "\n" >> $logfile
}

for imgpath in img_in/*
do
set -x
bench1d $imgpath
set +x
done