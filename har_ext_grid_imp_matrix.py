import pandapower as pp
import numpy as np
import math
import cmath

def impedance_matrix(network, har_order):
    # Used for transformation between sequence and phase systems
    a1 = cmath.rect(1, 2/3*cmath.pi)
    a2 = cmath.rect(1, 4/3*cmath.pi)
    
    matrix_A = np.matrix([[1, 1, 1], [1, a2, a1], [1, a1, a2]])
    
    z_ext_grid = {}
    
    z_pos_ohm = network.bus.vn_kv[network.ext_grid['bus'][0]]**2/network.ext_grid.s_sc_max_mva
    delta_pos = math.atan(1/network.ext_grid.rx_max)
    r_pos = z_pos_ohm*math.cos(delta_pos)
    
    z_zer_ohm = z_pos_ohm*network.ext_grid.x0x_max 
    delta_zer = math.atan(1/network.ext_grid.r0x0_max)
    r_zer = z_zer_ohm*math.cos(delta_zer)   
    
    for h in range(1, har_order + 1, 2):
        x_pos = z_pos_ohm*math.sin(delta_pos)*h
        z_pos = complex(r_pos, x_pos)
        
        x_zer = z_zer_ohm*math.sin(delta_zer)*h
        z_zer = complex(r_zer, x_zer)
        
        z_ext_grid_012_h = np.zeros([3,3], dtype = complex)
        
        z_ext_grid_012_h[0, 0] = z_zer/(network.bus.vn_kv[network.ext_grid['bus'][0]]**2)
        z_ext_grid_012_h[1, 1] = z_pos/(network.bus.vn_kv[network.ext_grid['bus'][0]]**2)
        z_ext_grid_012_h[2, 2] = z_pos/(network.bus.vn_kv[network.ext_grid['bus'][0]]**2)
        
        # print('0: ', abs(z_zer))
        
        z_ext_grid[h] = np.matmul(np.matmul(np.linalg.inv(matrix_A), z_ext_grid_012_h), matrix_A)
    
    return z_ext_grid
        