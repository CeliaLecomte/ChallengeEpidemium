#################################################################################################
#                    A-R-O-U-N-D trial project
#                     An EPIDEMIUM project
#                        Challenge 1-2
#          
#    Analyses done by: Elie Toledano, Aurélie Bardet
#    Date of Analysis: January 2018
#    Datasets: From score santé and E-cancer clinical trials registry


# Scope of the anylyses: Create viz for powerpoint presentation
# 
#################################################################################################

#Maps of cumulative expertise based on multigraphs function by Aurélie Bardet

require(dplyr)
require(maptools)
require(RColorBrewer)
require(classInt)
require(plotrix)

#definition of departments borders
departements<-readShapeSpatial("./DEPARTEMENT.SHP")

Mapsdata <- StudiesAll %>% filter(year == 2012)
Mapsdata$Department <- as.factor(Mapsdata$Department)
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="1"] <- "01"
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="2"] <- "02"
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="3"] <- "03"
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="5"] <- "05"
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="6"] <- "06"
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="7"] <- "07"
levels(Mapsdata$Department)[levels(Mapsdata$Department)=="8"] <- "08"


Compare <- match(departements$CODE_DEPT, Mapsdata$Department)

#number of classes
nclr <- 12
#definition of colors and classes
col <- findColours(classIntervals(Mapsdata$CumCount[Compare], style="fixed", fixedBreaks=c(0, 100, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500)),
                   smoothColors("white",98,"#0C3269"))
#definition of legend
leg <- findColours(classIntervals(
  round(Mapsdata$CumCount[Compare]), nclr, style="fixed", fixedBreaks=c(0, 100, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500)),
  smoothColors("white",3,"#0C3269"),
  under="<", over=">", between="–",
  cutlabels=FALSE)




#save graph
png(filename = "Studies map 2012 Cumulative.png", width = 1200, height = 760)
#plot
plot(departements,col=col)
#add of legend and title
title(main=paste("All Cancer","\n",sep=""),cex.main=1)
title(main=paste("\n \n Cumulative studies started","\n",sep=""),cex.main=1)
title(main=paste("\n \n \n \n","between 2002 and 2012",sep=""),cex.main=0.7)
legend(10000,6561753,fill=attr(leg, "palette"),
       legend=names(attr(leg,"table")),,cex=0.60)
dev.off() #end of graph save/close of plot function
rm(Mapsdata)
rm(Compare)
rm(nclr)
rm(col)
rm(leg)

#######


## Export graph for selected departments
#Selected department Paris(75), Rhônes-Alpes(69), Gironde (33), Nord(59), Bouches du Rhône(13), Bas-Rhin(67)

PlotStudies <- function(Study, pat, ylimits, out)
{

require(RColorBrewer)
require(dplyr)

col<-brewer.pal(6,"Paired")
x  <- 

s75 <- Study %>%  filter(Department == 75)
s69 <- Study %>%  filter(Department == 69)
s33 <- Study %>%  filter(Department == 33)
s59 <- Study %>%  filter(Department == 59)
s13 <- Study %>%  filter(Department == 13)
s67 <- Study %>%  filter(Department == 67)


png(filename = out, width = 1200, height = 760)
plot(s75$year, s75$CumCount,type="l",col=col[1],lwd=2,ylim=ylimits,lab = c(12, 15, 0),
     xlab = "Years", ylab = "Cumulative Studies started")


lines(s69$year,s69$CumCount,lty=1,col=col[2],lwd=2)
lines(s33$year,s33$CumCount,lty=1,col=col[3],lwd=2)
lines(s59$year,s59$CumCount,lty=1,col=col[4],lwd=2)
lines(s13$year,s13$CumCount,lty=1,col=col[5],lwd=2)
lines(s67$year,s67$CumCount,lty=1,col=col[6],lwd=2)

title(main=paste(pat,"\n",sep=""),cex.main=1)
title(main=paste("\n \n Cumulative Study started","\n",sep=""),cex.main=1)
title(main=paste("\n \n \n \n","between 2002 and 2012 in 6 departments",sep=""),cex.main=0.7)
legend("topleft", legend=c("Paris","Rhône", "Gironde", "Nord", "Bouches-du-Rhône", "Bas-Rhin"),
       col=col, lty=1, lwd = 3, cex=1)
dev.off()#####

}
 
StudiesAll %>% PlotStudies(pat = "All Cancer", ylimits = c(0,1500), out = "Studies cum per dpt - All Cancer.png")
StudiesSein %>% PlotStudies(pat = "Sein", ylimits = c(0,150), out = "Studies cum per dpt Breast.png")
StudiesPoumon %>% PlotStudies(pat = "Poumon", ylimits = c(0,150), out = "Studies cum per dpt Lung.png")
StudiesPancreas %>% PlotStudies(pat = "Pancreas", ylimits = c(0,40), out = "Studies cum per dpt Pancreas.png")
StudiesColorectal %>% PlotStudies(pat = "Colorectal", ylimits = c(0,170), out = "Studies cum per dpt Colorectal.png")
