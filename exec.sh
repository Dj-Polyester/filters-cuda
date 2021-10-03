for file in img_in/*
do
basefile=$(basename $file)
if [ "$#" -eq 2 ]
then
set -x
./build/main ./$file ./img_out/$basefile $@
set +x
elif [ "$#" -eq 3 ]
then
set -x
./build/main ./$file ./img_out/$basefile $@ 
set +x
fi
done