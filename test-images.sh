> output.txt
for object in $(seq 0 99)
do
    for angle in $(seq 0 5 355)
    do
        echo -ne "\r $object, $angle : " >> output.txt
        python test.py --test-single-image --object-id $object --angle $angle >> output.txt
    done
done    