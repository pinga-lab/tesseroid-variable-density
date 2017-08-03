#!/bin/bash

for file in ./*.svg; do
    inkscape --export-pdf ${file%.svg}.pdf ./$file
done
