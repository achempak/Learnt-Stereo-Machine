#!/bin/bash
URL=http://people.eecs.berkeley.edu/~akar/lsm/models_lsm_v1_small.tar.gz
CHECKSUM=8955572bb300a52a8b0dcf78d7e47b19
FILE=/home/adityac/Downloads/models_lsm_v1_small.tar.gz

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

#wget $URL -O $FILE
tar xvzf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
