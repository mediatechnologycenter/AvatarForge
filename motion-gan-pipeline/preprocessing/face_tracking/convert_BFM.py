import numpy as np
from scipy.io import loadmat

original_BFM = loadmat('3DMM/01_MorphableModel.mat')
sub_inds = np.load('3DMM/topology_info.npy',
                   allow_pickle=True).item()['sub_inds']

print(len(sub_inds))

shapePC = original_BFM['shapePC']
shapeEV = original_BFM['shapeEV']
shapeMU = original_BFM['shapeMU']
texPC = original_BFM['texPC']
texEV = original_BFM['texEV']
texMU = original_BFM['texMU']

print('shapePC: ', shapePC.shape)
print('shapeMU: ', shapeMU.shape)
print('texPC: ', texPC.shape)
print('texMU: ', texMU.shape)


b_shape = shapePC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_shape = shapeMU.reshape(-1, 3)
print('\nshapePC -> b_shape : ', b_shape.shape)
print('shapeMU -> mu_shape : ', mu_shape.shape)

b_shape = b_shape[:, sub_inds, :].reshape(199, -1)
mu_shape = mu_shape[sub_inds, :].reshape(-1)
print('b_shape -> b_shape sampled : ', b_shape.shape)
print('mu_shape -> mu_shape sampled: ', mu_shape.shape)

b_tex = texPC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_tex = texMU.reshape(-1, 3)
print('\ntexPC -> b_tex : ', b_tex.shape)
print('texMU -> mu_tex : ', mu_tex.shape)
b_tex = b_tex[:, sub_inds, :].reshape(199, -1)
mu_tex = mu_tex[sub_inds, :].reshape(-1)
print('b_tex -> b_tex sampled : ', b_tex.shape)
print('mu_tex -> mu_tex sampled: ', mu_tex.shape)

exp_info = np.load('3DMM/exp_info.npy', allow_pickle=True).item()
print('\nexp_info: ')
for key in exp_info.keys():
    print(f'{key} shape: {exp_info[key].shape}')

info = {'mu_shape': mu_shape,
        'b_shape': b_shape,
        'sig_shape': shapeEV.reshape(-1),
        'mu_exp': exp_info['mu_exp'],
        'b_exp': exp_info['base_exp'],
        'sig_exp': exp_info['sig_exp'],
        'mu_tex': mu_tex,
        'b_tex': b_tex,
        'sig_tex': texEV.reshape(-1)}

print('\nFinal: ')
for key in info.keys():
    print(f'{key} shape: {info[key].shape}')

np.save('3DMM/3DMM_info.npy', info)
