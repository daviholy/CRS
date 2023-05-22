#!/bin/sh
cat=$(ls ./czech_text_document_corpus_v20/*.tok | cut -d '/' -f 3 | cut -d '.' -f 1 | cut -d "_" -f 2- | tr "_" "\n" | sort | uniq -c | sort -n |tail -n 10 | sed 's/^ *//' | cut -d ' ' -f 2 | tr "\n" "|")

files=$(ls ./czech_text_document_corpus_v20/*.lemma | grep -E ${cat} | tr " " "\n" )

for file in ${files}; do
  file_name=$(ls ${file} | cut -d "/" -f 3 )
  cp ${file} ./vyber/${file_name}
done
