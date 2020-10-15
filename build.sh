#!/bin/bash

basepath=$(cd `dirname $0`; pwd)
make
grep "tmtool" /etc/bash.bashrc
if [ $? -eq 0 ] ;then
   sed -i '$d' /etc/bash.bashrc
   sed -i '$a\alias tmtool=\'\'${basepath}'/out/tmtool'\' /etc/bash.bashrc
   echo "tmtool bash have been updated !"
else
  sed -i '$a\alias tmtool=\'\'${basepath}'/out/tmtool'\' /etc/bash.bashrc
fi
