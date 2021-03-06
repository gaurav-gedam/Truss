# Read an existing msh file and prepare element topology data arrays

def oneDmesh():
    import gmsh
    gmsh.initialize()
    gmsh.open('mesh.msh')
    
    import numpy as np
    # read dodal data 
    nodetags = np.array(gmsh.model.mesh.getNodes(-1,-1)[0])
    coords = np.array(gmsh.model.mesh.getNodes(-1,-1)[1])
    
    coord_indices = np.array([*range(nodetags.size)], dtype=int)
    
    x = np.array(coords[3*coord_indices])
    y = np.array(coords[3*coord_indices+1])
    z = np.array(coords[3*coord_indices+2])
    
    # read element data
    
    elemtags = np.array(gmsh.model.mesh.getElements(dim=1,tag=-1)[1]).flatten()
    elem_nodetags = np.array(gmsh.model.mesh.getElements(dim=1,tag=-1)[2]).flatten()
    node_indices = np.array([*range(elemtags.size)], dtype=int)
    
    n1 = np.array(elem_nodetags[2*node_indices])
    n2 = np.array(elem_nodetags[2*node_indices+1])
    
    return(nodetags,x,y,z,elemtags,n1-1,n2-1)
