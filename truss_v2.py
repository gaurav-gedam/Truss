# Program for finite element simulation of 3D truss.
# Author: Gaurav Gedam 
# Element Type: 2-node continuum line element
# Physics: 'Pin jointed' general 3D truss structure. Elements support axial forces and displacements.
# Input: Meshed model saved as 'Mesh.msh', gmsh V2 is used
#        material and cross section data in 'mat.dat'
#        Point load data in 'ptload.dat'
#        BC data in bc.dat
# Output: Nodal displacements, axial forces, axial stresses, total strain energy

import numpy as np
from oneDmesh import oneDmesh

# generate mesh from geometry and read mesh data
nodetags, x, y, z, elemtags, n1, n2 = oneDmesh()
numnp = x.size                  # total number of nodes
numel = n1.size               # total number of elements                  
n = 3 * numnp                 # total number of dof's

# Read material and area data
E, A = np.loadtxt("mat.dat", unpack=True)

# Convert signle entry into np array
if E.size == 1: E = np.array([E])
if A.size == 1: A = np.array([A])

# Read element data 
matset, areaset = np.zeros((numel),dtype=int), np.zeros((numel),dtype=int)

# Form local coordinate arrays
x1 = x[n1]
y1 = y[n1]
z1 = z[n1]
x2 = x[n2]
y2 = y[n2]
z2 = z[n2]

# Lenghts of truss elements
dx = x2-x1
dy = y2-y1
dz = z2-z1
L = np.sqrt(dx**2 + dy**2 + dz**2)

# Direction cosines
cx = dx/L
cy = dy/L
cz = dz/L

# Axial stiffness array
ax_stiff = E[matset] * A[areaset]/L

# Element stiffness matrices
km = np.array([[cx**2, cx*cy, cx*cz, -cx**2, -cx*cy, -cx*cz],
               [cx*cy, cy**2, cy*cz, -cx*cy, -cy**2, -cy*cz],
               [cx*cz, cy*cz, cz**2, -cx*cz, -cy*cz, -cz**2],
               [-cx**2, -cx*cy, -cx*cz, cx**2, cx*cy, cx*cz],
               [-cx*cy, -cy**2, -cy*cz, cx*cy, cy**2, cy*cz],
               [-cx*cz, -cy*cz, -cz**2, cx*cz, cy*cz, cz**2]])*ax_stiff

# Global stiffness matrix assembly
K = np.zeros((n, n))
for h in range(numel):
    LM = np.array([3*n1[h], 3*n1[h]+1, 3*n1[h]+2, 3*n2[h], 3*n2[h]+1, 3*n2[h]+2], dtype=int)
    for i in range(6):
        for j in range(6):
            K[LM[i], LM[j]] += km[i,j,h]

# Read point loads
ptnode = np.loadtxt('ptload.dat',unpack=True,usecols=(0), dtype=int)
fx, fy, fz = np.loadtxt('ptload.dat',unpack=True,usecols=(1,2,3))

# Convert signle entry into np array
if ptnode.size == 1: 
    ptnode = np.array([ptnode])
    fx = np.array([fx])
    fy = np.array([fy])
    fz = np.array([fz])

# Generate load vector
R = np.zeros((n,1))
for c in range(ptnode.size):
    R[np.array([3*ptnode[c], 3*ptnode[c]+1, 3*ptnode[c]+2])] = [[fx[c]], [fy[c]], [fz[c]]]
    
# Read BC data
bcnodes, dirn = np.loadtxt("bc.dat", unpack=True, dtype=int, usecols=(0,1))
disp = np.loadtxt("bc.dat", unpack=True, usecols=(2))

# Modify load vector to apply Dirichlet BC's
for i in range(bcnodes.size):
    R -= K[:,3*bcnodes[i]+dirn[i]][:,None] * disp[i]

# Make an array of active dof numbers
adof = np.delete(np.array(range(3*numnp)), np.array([3*bcnodes+dirn], dtype=int))

################################### Solve for unknown displacements ###########

# Solve for displacements
# With sparse solver
# from scipy.sparse.linalg import spsolve
# u = spsolve(K[np.ix_(adof,adof)], R[np.ix_(adof)])

# With general solver
from scipy.linalg import solve
u = solve(K[np.ix_(adof,adof)], R[np.ix_(adof)])

##############################################################################
#                               Post-processing
##############################################################################

# Global displacement vector
disp_vec = np.zeros((n)) 
disp_vec[adof] = u.flatten()
disp_vec[3*bcnodes+dirn] = disp

# nodal_force (for validation)
nodal_force = K@disp_vec

# Total strain energy stored in the structure
U = nodal_force @disp_vec*0.5

# Node locations after deformation
xdef = x + disp_vec[np.array(range(0,3*numnp-1,3))]
ydef = y + disp_vec[np.array(range(1,3*numnp-1,3))]
zdef = z + disp_vec[np.array(range(2,3*numnp,3))]

xdef1 = xdef[n1]
ydef1 = ydef[n1]
zdef1 = zdef[n1]
xdef2 = xdef[n2]
ydef2 = ydef[n2]
zdef2 = zdef[n2]

# Axial force and axial stress in bars
L2 = np.sqrt((xdef2-xdef1)**2 + (ydef2-ydef1)**2 + (zdef2-zdef1)**2)

axforce = A*E*(L2/L-1) # Forces along member axes
axstress = axforce*A   # Stresses along member axes

#____________________________________________ POST PROCESS IN PARAVIEW__________________
# Write .vtk file and open it in Paraview
# create result file and write the header
resundef = open("res_undef.vtk", "w") #undeformed geometry
resdef = open("res_def.vtk", "w") #deformed geometry
header = ["# vtk DataFile Version 3.0\n", "TRUSS\n", "ASCII\n", "DATASET UNSTRUCTURED_GRID\n"]
resundef.writelines(header)
resdef.writelines(header)

# write nodal point data
resundef.writelines(["POINTS ", str(numnp), " FLOAT\n"])
resdef.writelines(["POINTS ", str(numnp), " FLOAT\n"])

for i in range(numnp):
    resundef.writelines([str(x[i]), " ", str(y[i]), " ", str(z[i]), "\n"]) #Undeformed nodes
    resdef.writelines([str(xdef[i]), " ", str(ydef[i]), " ", str(zdef[i]), "\n"]) #Deformed nodes

# write element connectivity data
resundef.writelines(["CELLS ", str(numel), " ", str(3*numel), "\n"])
resdef.writelines(["CELLS ", str(numel), " ", str(3*numel), "\n"])
for i in range(numel):
    resundef.writelines([str(2), " ", str(n1[i]), " ", str(n2[i]), "\n"]) #Undefomed mesh
    resdef.writelines([str(2), " ", str(n1[i]), " ", str(n2[i]), "\n"]) #Defomed mesh
    
# write paraview cell type used(here 2 noded line i.e VTK_LINE is used)
resundef.writelines(["CELL_TYPES ", str(numel), "\n"])
resdef.writelines(["CELL_TYPES ", str(numel), "\n"])
resundef.write(numel * "3\n")
resdef.write(numel * "3\n")

# write nodal displacements
resundef.writelines(["POINT_DATA ", str(numnp), "\n", "VECTORS NODAL_DISPACEMENTS FLOAT\n"])
resdef.writelines(["POINT_DATA ", str(numnp), "\n", "VECTORS NODAL_DISPACEMENTS FLOAT\n"])
for i in range(numnp):
    resundef.writelines( [str(disp_vec[3*i+0]), " ", str(disp_vec[3*i+1]), " ", str(disp_vec[3*i+2]), "\n"] )
    resdef.writelines( [str(disp_vec[3*i+0]), " ", str(disp_vec[3*i+1]), " ", str(disp_vec[3*i+2]), "\n"] )
    
# write element forces
resundef.writelines(["CELL_DATA ", str(numel), "\n", "SCALARS INTERNAL_FORCE FLOAT\n", "LOOKUP_TABLE DEFAULT\n"])
resdef.writelines(["CELL_DATA ", str(numel), "\n", "SCALARS INTERNAL_FORCE FLOAT\n", "LOOKUP_TABLE DEFAULT\n"])
for i in range(numel):
    resundef.writelines( [str(axforce[i]), "\n" ] )
    resdef.writelines( [str(axforce[i]), "\n" ] )

resundef.close()
resdef.close()

# import os
#os.startfile('res_undef.vtk')
#os.startfile('res_def.vtk')

np.savetxt('out.dat', u)