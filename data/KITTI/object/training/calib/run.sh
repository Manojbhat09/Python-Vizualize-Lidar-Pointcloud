for ((i=1; i<=153; i=i+1))
do 
	printf -v j "%04d.txt" $i
	cp 0000.txt $j
done
