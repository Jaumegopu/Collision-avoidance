#function to evaluate 

def f_action(zone,brightness,action): #inputs: Zone I - VI. Brightness==distance. 
    val=0
    #2m==200, 1.2m==140
    
    if zone==1: #Central Zone
        if brightness<140: #danger zone 
            if action==0 or action==1 or action ==5 or action==6 or action==7:
                val=val-1
            elif action==3 or action==4:
                val=val+1
            else:
                val=val
        elif brightness>200: #comfort zone
            if action==0 or action==1 or action ==5 or action==6 or action==7:
                val=val+1
            elif action==3 or action==4:
                val=val-1
            else:
                val=val
            
        else: #safe zone
            if action==0 :
                val=val-1
            elif action==3 or action==4 or action==6 or action==7:
                val=val+1
            else:
                val=val

    elif zone==3: #Right zone
        if action==0 or action==1 or action ==5 or action==7:
            val=val+1
        elif action==4 or action==6: #penalitzem gir dreta
            val=val-1
        else:
            val=val

    elif zone==2: #Left zone
        if action==0 or action==1 or action ==5 or action==6:
            val=val+1
        elif action==3 or action==7: #penalitzem gir esquerra
            val=val-1
        else:
            val=val
    else: 
        print('Unvalid pixel zone')
    return val


zone=1
brightness=201
action=0
val=f_action(zone,brightness,action)
print(val)
