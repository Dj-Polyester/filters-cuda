for file in img_in/*
do
basefilename=$(basename $file)
if [ "$#" -eq 2 ]
then
set -x
logfile="logs/$1_$2.log"
echo $basefilename >> $logfile
./build/main ./$file ./img_out/$basefilename $@ 2>> $logfile
set +x
elif [ "$#" -eq 3 ]
then
set -x
logfile="logs/$1_$2_$3.log"
echo $basefilename >> $logfile
./build/main ./$file ./img_out/$basefilename $@ 2>> $logfile 
set +x
fi
done