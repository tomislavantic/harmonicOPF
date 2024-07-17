import random as rd

def harmonic_current(loads):
    
    harmonic_current_a = {}
    harmonic_current_b = {}
    harmonic_current_c = {}
    
    for i in loads:
    
        harmonic_current_a[3,i] = 15/100*rd.uniform(0.7, 1.3)
        harmonic_current_b[3,i] = 15/100*rd.uniform(0.7, 1.3)
        harmonic_current_c[3,i] = 15/100*rd.uniform(0.7, 1.3)
        
        harmonic_current_a[5,i] = 13.5/100*rd.uniform(0.7, 1.3)
        harmonic_current_b[5,i] = 13.5/100*rd.uniform(0.7, 1.3)
        harmonic_current_c[5,i] = 13.5/100*rd.uniform(0.7, 1.3)

        harmonic_current_a[7,i] = 12/100*rd.uniform(0.7, 1.3)
        harmonic_current_b[7,i] = 12/100*rd.uniform(0.7, 1.3)
        harmonic_current_c[7,i] = 12/100*rd.uniform(0.7, 1.3)
        
        harmonic_current_a[9,i] = 10/100*rd.uniform(0.7, 1.3)
        harmonic_current_b[9,i] = 10/100*rd.uniform(0.7, 1.3)
        harmonic_current_c[9,i] = 10/100*rd.uniform(0.7, 1.3)

        harmonic_current_a[11,i] = 6.5/100*rd.uniform(0.7, 1.3)
        harmonic_current_b[11,i] = 6.5/100*rd.uniform(0.7, 1.3)
        harmonic_current_c[11,i] = 6.5/100*rd.uniform(0.7, 1.3)
            
    return harmonic_current_a, harmonic_current_b, harmonic_current_c