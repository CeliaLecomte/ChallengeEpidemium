#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Script File: coreDataset2R.r 
# Date of creation: 3 mars 2016
# Date of last modification: 3 mars 2016
# Author: Seraya Maouche <seraya.maouche@iscb.org>
# Project: EpidemiumDB (http://www.epidemium.cc)
# Short Description: This script provides functionalities to import, QC, and export
#                                cancer incidences and mortality from the core dataset
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Download the core dataset from data.epidemium.cc
# http://data.epidemium.cc/files/core/

```R
# Cancer incidences data
setwd("../BD4Cancer/data")
dat <- read.csv("incidence_1953_2007.csv", header=T, sep=",")
header(dat)
datFrance <- dat[grep("^France", dat$registry), ]
write.table(datFrance,"incidenceCanFrance.txt", sep = "\t")
save(dat, file = "incidencesAllCoreDataset.RData")
save(datFrance, file = "incidenceCancFrance.RData")
```

Mortality data are also in the same file.

```R
# Mortality data
datadir   <- "/home/common/data/" 
filename  <-"mortalite_1950_2013.csv"
setwd(datadir)
mortality <- read.csv(filename, header = TRUE, sep=";") 
dim(mortality) 
[1] 3191598       6 
head(mortality)    
id.cancer  sex age registry year cases.number
1        C11 Male 0-4  Albania 2009            0
2     C00-14 Male 0-4  Albania 2009            0
3 C00-15,C32 Male 0-4  Albania 2009            0
4        C15 Male 0-4  Albania 2009            0
5        C16 Male 0-4  Albania 2009            0
6     C17-21 Male 0-4  Albania 2009            0 
# Subset using the registry column to extract data for France
dataFrance <- subset(mortality, mortality[,4] =="France") 
dim(dataFrance)
[1] 59850     6
```
