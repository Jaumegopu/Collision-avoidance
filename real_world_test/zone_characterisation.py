#determinar a quina zona està l'objecte més proper
#dos linies: (0,97) a (55,74) i (160,97) a (105,74)

def zone_det(pixel):
    f,c=pixel #fila,columna
    f=128-f #we invert the row index as the image has the 0 at the top and the 128 at the bottom
    if c<55 and f>23*c/55 + 31:
        zone=2
    elif c>105 and f>-23*c/55+97:#l'equacio no està bé
        zone=3
    else:
        zone=1
    return zone
    

"""
pixel=[76,106]
zone=zone_det(pixel)
print('Zone:', zone)
"""