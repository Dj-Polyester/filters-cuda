imgbasename=$1
imgname="${imgbasename%%.*}"
imgext="${imgbasename#*.}"

logpath="benchmarks/$imgbasename.log"

echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_mean.${imgext} mean 32 32 12)" > $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_gaussian.${imgext} gaussian 32 32 12 25)" >> $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_meanSeparable.${imgext} meanSeparable 32 32 12)" >> $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_gaussianSeparable.${imgext} gaussianSeparable 32 32 12 25)" >> $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_meanShared.${imgext} meanShared 32 32 12)" >> $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_gaussianShared.${imgext} gaussianShared 32 32 12 25)" >> $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_meanSharedSep.${imgext} meanSharedSep 32 32 12)" >> $logpath
echo "$imgbasename: $(./debug2.sh $imgbasename ${imgname}_gaussianSharedSep.${imgext} gaussianSharedSep 32 32 12 25)" >> $logpath
