#!/bin/sh

cd /data
echo 'move into /data directory'

# download tiny imagent dataset
WGET=/usr/bin/wget
URL='http://cs231n.stanford.edu/tiny-imagenet-200.zip'

$WGET $URL

echo 'Download done [*]'

