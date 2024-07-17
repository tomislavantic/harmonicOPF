import pandapower as pp
import pandas as pd
import unbalanced_hopf

net = pp.from_excel(r'network.xlsx')

df_p1 = pd.read_excel(r'p1_hour.xlsx', header = 0, index_col= 0)
df_p2 = pd.read_excel(r'p2_hour.xlsx', header = 0, index_col= 0)
df_p3 = pd.read_excel(r'p3_hour.xlsx', header = 0, index_col= 0)

df_pv_a = pd.read_excel(r'pv_2_a.xlsx', header = 0, index_col= 0)
df_pv_b = pd.read_excel(r'pv_2_b.xlsx', header = 0, index_col= 0)
df_pv_c = pd.read_excel(r'pv_2_c.xlsx', header = 0, index_col= 0)

obj_val, v_a, v_b, v_c, vuf, thd_a, thd_b, thd_c, p_gen_a, p_gen_b, p_gen_c, q_gen_a, q_gen_b, q_gen_c \
    = unbalanced_hopf.hopf_3ph(net, 7, df_p1, df_p2, df_p3, list(net.asymmetric_load.bus.values), df_pv_a, df_pv_b, df_pv_c, 1.0)

v_a.to_excel(r'Scenario 2\v_a_m3.xlsx')
v_b.to_excel(r'Scenario 2\v_b_m3.xlsx')
v_c.to_excel(r'Scenario 2\v_c_m3.xlsx')

vuf.to_excel(r'Scenario 2\vuf_m3.xlsx')

thd_a.to_excel(r'Scenario 2\thd_a_m3.xlsx')
thd_b.to_excel(r'Scenario 2\thd_b_m3.xlsx')
thd_c.to_excel(r'Scenario 2\thd_c_m3.xlsx')

p_gen_a.to_excel(r'Scenario 2\p_gen_a_m3.xlsx')
p_gen_b.to_excel(r'Scenario 2\p_gen_b_m3.xlsx')
p_gen_c.to_excel(r'Scenario 2\p_gen_c_m3.xlsx')

q_gen_a.to_excel(r'Scenario 2\q_gen_a_m3.xlsx')
q_gen_b.to_excel(r'Scenario 2\q_gen_b_m3.xlsx')
q_gen_c.to_excel(r'Scenario 2\q_gen_c_m3.xlsx')

print(1000*obj_val)
