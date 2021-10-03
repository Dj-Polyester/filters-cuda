
WARPSIZE=32
MAXBLOCKSIZE=1024
filterfunc=$1

bench1d() {
imgpath=$1

imgbasename=$(basename $imgpath)

imgname="${imgbasename%%.*}"
imgext="${imgbasename#*.}"

logfile="logs/$filterfunc.log"

echo $imgbasename >> $logfile

for ((blocksize = $WARPSIZE ; blocksize <= $MAXBLOCKSIZE ; blocksize+=$WARPSIZE))
do
printf "$blocksize " >> $logfile
done
printf "\n" >> $logfile

for ((blocksize = $WARPSIZE ; blocksize <= $MAXBLOCKSIZE ; blocksize+=$WARPSIZE))
do
newimg="${imgname}_${filterfunc}_${blocksize}.${imgext}"
val=$(./build/main ./$imgpath ./img_out/$newimg $filterfunc $blocksize)
printf "$val " >> $logfile
done
printf "\n" >> $logfile
}

for imgpath in img_in/*
do
set -x
bench1d $imgpath
set +x
done