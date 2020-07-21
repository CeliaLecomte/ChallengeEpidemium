#################################################################################################
#                    A-R-O-U-N-D trial project
#                     An EPIDEMIUM project
#                        Challenge 1-2
#          
#    Analyses done by: Elie Toledano, Aurelie Bardet
#    Date of Analysis: January 2018
#    Datasets: from SCORE SANTE website
#    Aim of the program: first univariate analyses on deaths
#
#################################################################################################

#installing packages
install.packages("RColorBrewer",dependencies=TRUE)
install.packages("classInt",dependencies=TRUE)
install.packages("plotrix",dependencies=TRUE)

require(RColorBrewer)
require(classInt)
require(plotrix)

#recovery of data
deaths<-read.table("./final_table_for_death.csv",
                 header = TRUE, sep = ",", row.names = 1)

###presentation of data -> cartography
#based on program carto.R by Elie Toledano
#source=http://coulmont.com/cartes/rcarto.pdf, paragraph 2.2, maptools and shapefile from ign

require(maptools)

#definition of departments borders
departements<-readShapeSpatial("./DEPARTEMENT.SHP")
summary(departements)
plot(departements)

multigraphs<-function(pat,breaks,nclass,ylimits,out1,out2)
{

#selection of data to represent
data0<-subset(deaths,YEAR==2012)
data00<-subset(data0,PATHOLOGY==pat)
data1<-subset(data00,GENDER=="All")

#recovery of geographical latitudes and longitudes by department / all information in one database
#departements$CODENUM<-as.numeric(departements$CODE_DEPT)
#d<-subset(departements,NOM_DEPT=="AIN")
#summary(d)
allinfo<-merge(data1,departements,by.x="DEPARTMENT",by.y="CODE_DEPT")

#graphical representation
require(RColorBrewer)
require(classInt)
require(plotrix)
#number of classes
nclr <- nclass
#definition of colors and classes
col <- findColours(classIntervals(allinfo$DEATHRATE, style="fixed",fixedBreaks=breaks),
  smoothColors("#0C3269",98,"white"))
#definition of legend
leg <- findColours(classIntervals(
  round(allinfo$DEATHRATE), nclass, style="fixed",fixedBreaks=breaks),
  smoothColors("#0C3269",3,"white"),
  under="<", over=">", between="–",
  cutlabels=FALSE)

#save graph
png(filename = out1, width = 480, height = 480)
#plot
plot(departements,col=col)
#add of legend and title
title(main=paste(pat,"\n",sep=""),cex.main=1)
title(main=paste("\n \n Standardized mortality rate in 2012","\n",sep=""),cex.main=1)
title(main=paste("\n \n \n \n","per 100 000 inhabitants",sep=""),cex.main=0.7)
legend(40000,6561753,fill=attr(leg, "palette"),
       legend=names(attr(leg,"table")),,cex=0.60)
dev.off() #end of graph save/close of plot function


###graphical representation of time evolution in selected departments
data99<-subset(deaths,PATHOLOGY==pat)
data999<-subset(data99,GENDER=="All")
selectedinfo<-subset(data999,DEPARTMENT %in% c(67,75,59,33,69,13,44))
#Bas-Rhin (Strasbourg 67), Paris 75, Nord (Lille 59), Gironde (Bordeaux 33), 
#Rhône (Lyon 69), Bouches-du-Rhône (Marseille 13), Loire-Atlantique (Nantes 44)

x  <- c(2002:2012)
y67 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(67))
y75 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(75))
y59 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(59))
y33 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(33))
y69 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(69))
y13 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(13))
y44 <- subset(selectedinfo$DEATHRATE,selectedinfo$DEPARTMENT %in% c(44))

#definition of colors
display.brewer.all(type="qual") 
col<-brewer.pal(6,"Paired")

#save graph
png(filename = out2, width = 480, height = 480)
plot(x,y75,type="l",col=col[1],lwd=2,ylim=ylimits,lab = c(12, 15, 0),
     xlab = "Years", ylab = "Mortality rate")
lines(x,y69,lty=1,col=col[2],lwd=2)
lines(x,y33,lty=1,col=col[3],lwd=2)
lines(x,y59,lty=1,col=col[4],lwd=2)
lines(x,y13,lty=1,col=col[5],lwd=2)
lines(x,y67,lty=1,col=col[6],lwd=2)
title(main=paste(pat,"\n",sep=""),cex.main=0.9)
title(main=paste("\n \n Standardized mortality rate","\n",sep=""),cex.main=0.8)
title(main=paste("\n \n \n \n","per 100 000 inhabitants",sep=""),cex.main=0.6)
title(main=paste("\n \n \n \n \n \n","Selected regions",sep=""),cex.main=0.6)
legend("bottomleft", legend=c("Paris", "Rhône","Gironde","Nord","Bouches-du-Rhône","Bas-Rhin"),
       col=col, lty=1, cex=0.8)
dev.off()

}
sortie<-multigraphs(pat="All",breaks=c(0,220,230,240,250,260,270,280,290,300,100000),nclass=10,ylimits=c(210,350),
  out1="Death map_2012_All.png",out2="Death graph_All_Selected.png")
sortie<-multigraphs(pat="Breast",breaks=c(0,15,17.5,20,22.5,25,27.5,100000),nclass=7,ylimits=c(13,32),
                    out1="Death map_2012_Breast.png",out2="Death graph_Breast_Selected.png")
sortie<-multigraphs(pat="Pancreas",breaks=c(0,14,16,18,20,22,100000),nclass=6,ylimits=c(8,22),
                    out1="Death map_2012_Pancreas.png",out2="Death graph_Pancreas_Selected.png")
sortie<-multigraphs(pat="Colon",breaks=c(0,15,20,25,30,35,40,100000),nclass=7,ylimits=c(20,40),
                    out1="Death map_2012_Colon.png",out2="Death graph_Colon_Selected.png")
sortie<-multigraphs(pat="Lung",breaks=c(0,35,40,45,40,55,60,100000),nclass=7,ylimits=c(34,66),
                    out1="Death map_2012_Lung.png",out2="Death graph_Lung_Selected.png")

#end of program
