for logfile in logs/*.log
do 
baselogfile=$(basename $logfile)
logfilenoext="${baselogfile%%.*}"
printf "$logfilenoext "
done
echo ""