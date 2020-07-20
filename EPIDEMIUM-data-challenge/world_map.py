import numpy as np
import matplotlib as plt
import pygame as pygame


#import imageio
#im=imageio.imread('52955.png')
import matplotlib.pyplot as plt


from PIL import Image
im=Image.open('carte2.png')
#im.show()
tab=np.array(im)
#im=Image.fromarray(im)
#im.show()
d={}

n,l=tab_totale.shape
liste_pays=[]
for i in range(n):
    if(not tab_totale[i,0] in liste_pays):
        liste_pays.append(tab_totale[i,0])
print(liste_pays)

n=len(liste_pays)


d['Albania']=(274,930)
d['Argentina']=(800,440)
d['Armenia']=(277,1069)
d['Australia']=(736,1605)
d['Austria']=(227,901)
d['Azerbaijan']=(278,1084)
d['Belarus']=(190,964,)
d['Belgium']=(207,846)

liste=[279,437,513,682,958,260,269,166,402,736,358,536,307,487,909,241,357,415,898,213,870,172,390,426,334,566,274,461,954,156,946,130,829,228,1057,264,873,207,940,282,265,453,921,232,739,118,781,192,1020,343,889,262,1609,306,1175,224,1175,224,1093,353,953,167,945,178,853,213,897,313,200,390,1744,868,298,465,871,138,319,499,456,707,353,613,1545,439,922,197,774,276,954,238,928,242,902,238,958,771,792,273,471,529,899,141,862,234,1231,293,1382,407,442,485,1001,220,813,195,487,782]

liste2=['Belize', 'Brazil', 'Bulgaria', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Guatemala', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Kazakhstan', 'Kuwait', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Mauritius', 'Mexico', 'New Zealand', 'Nicaragua', 'Norway', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Serbia',  'Slovenia', 'South Africa', 'Spain', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 'Trinidad and Tobago', 'Ukraine', 'United Kingdom', 'Uruguay']

for i in range(len(liste2)):
    d[liste2[i]]=(liste[2*i+1],liste[2*i])
    
    
liste_pays=liste2+['Albania','Argentina','Armenia','Australia','Austria','Azerbaijan','Belarus','Belgium']

def coloriage():
    for pays in liste_pays:
        if(not pays =='Singapore'):
            print(pays)
            x,y=d[pays]
            couleur=color2(d2[pays],maxi,mini)
            print(couleur)
            attente=[(x,y)]
            VISITE=[(x,y)]
            while(attente!=[]):
                (x,y)=attente[0]
                del(attente[0])
                (tab[x,y,0],tab[x,y,1],tab[x,y,2])=couleur
                if(x+1<993 and tab[x+1,y,0]==255 and not (x+1,y) in VISITE):
                    attente.append((x+1,y))
                    VISITE.append((x+1,y))
                if(x-1>0 and tab[x-1,y,0]==255 and not (x-1,y) in VISITE):
                    attente.append((x-1,y))
                    VISITE.append((x-1,y))
                if(y+1<2000 and tab[x,y+1,0]==255 and not (x,y+1) in VISITE):
                    attente.append((x,y+1))
                    VISITE.append((x,y+1))
                if(y-1>0 and tab[x,y-1,0]==255 and not (x,y-1) in VISITE):
                    attente.append((x,y-1))
                    VISITE.append((x,y-1))
                
    return(tab)
    
import math as ma 

def color(x,maxi,mini):
    if(x<=0):
        ampli=mini
        return((200-ma.sqrt(ma.sqrt(x/ampli))*50,200+ma.sqrt(ma.sqrt(x/ampli))*50,0))
    else:
        ampli=maxi
        return((200+ma.sqrt(ma.sqrt(x/ampli))*50,200-ma.sqrt(ma.sqrt(x/ampli))*50,0))
    

def f(y):
    return(min(1,y))
def color2(x,maxi,mini):
    if(x<=0):
        ampli=mini
        return((255-(x/ampli)*255,255,0))
    else:
        ampli=maxi
        return((255,255-f(x)*255,0))
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    