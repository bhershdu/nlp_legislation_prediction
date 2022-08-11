#!/bin/bash
# "An Act" is a broad keyword 
# "Section" is a broad keyword
for f in *.html;do
	count=$(grep Section $f | wc -l)
	if [[ $count -ne 0 ]];then
		echo $f >> files-with-section.txt
	else
		echo $f >> files-without-sections.txt
	fi
done
