
WARPSIZE=32
num=$WARPSIZE

while [ $num -lt 544 ]
do
set -x
./exec.sh $1 $num
set +x
((num+=$WARPSIZE))
done