from scipy.spatial.distance import cdist
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import TETB_GRAPHENE.descriptors as descriptors
from sympy import *


#########################################################################################

# UTILS

########################################################################################

#@njit
def moon(r, a, b, c): 
    """
    Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012)
    """
    d, dz = r 
    return a * np.exp(-b * (d - 2.68))*(1 - (dz/d)**2) + c * np.exp(-b * (d - 6.33)) * (dz/d)**2
#@njit
def fang(rvec, a0, b0, c0, a3, b3, c3, a6, b6, c6, d6):
    """
    Parameterization from Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)
    """
    r, theta12, theta21 = rvec
    r = r / 4.649 

    def v0(x, a, b, c): 
        return a * np.exp(-b * x ** 2) * np.cos(c * x)

    def v3(x, a, b, c): 
        return a * (x ** 2) * np.exp(-b * (x - c) ** 2)  

    def v6(x, a, b, c, d): 
        return a * np.exp(-b * (x - c)**2) * np.sin(d * x)

    f =  v0(r, a0, b0, c0) 
    f += v3(r, a3, b3, c3) * (np.cos(3 * theta12) + np.cos(3 * theta21))
    f += v6(r, a6, b6, c6, d6) * (np.cos(6 * theta12) + np.cos(6 * theta21))
    return f

#@njit
def chebval(x, c):
    """Evaluate a Chebyshev series at points x."""
    c = np.asarray(c)
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x

#@njit
def norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in np.arange(a.shape[0]):
        sum=0
        for j in np.arange(a.shape[1]):
            sum += a[i,j]*a[i,j]
        norms[i] = np.sqrt(sum)
    return norms

###############################################################################

# POPOV

###############################################################################
#@njit
def popov_hopping(dR,params=None):
    """pairwise Slater Koster Interlayer hopping parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    aa = 1.0  # [Bohr radii]
    b = 10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    if params is None:
        Cpp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,
                            -0.0978079, 0.0577363, -0.0262833, 0.0094388,
                                -0.0024695, 0.0003863])
        Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,
                            -0.0535682, 0.0181983, -0.0046855, 0.0007303,
                          0.0000225, -0.0000393])
    else:
        Cpp_sigma = params[0,:]
        Cpp_pi = params[1,:]
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat*eV_per_hart

def popov_overlap(dR,params=None):
    """pairwise Slater Koster Interlayer overlap parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Overlap matrix elements [eV]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 10)
  
    #%boundaries for polynomial

    aa = 1 #; %Angstrom
    b = 10 #; %Angstrom
    y = (2*r-(b+aa))/(b-aa)
    #orignally sigma
    if params is None:
        Cpp_sigma=np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997,
                            0.0921727, -0.0268106, 0.0002240, 0.0040319,
                            -0.0022450, 0.0005596])
        Cpp_pi=   np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,
                            0.0156376, 0.0025976, -0.0039498, 0.0020581,
                            -0.0007114, 0.0001427])
    else:
        Cpp_sigma = params[0,:]
        Cpp_pi = params[1,:]

    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi  #; %Changing only this as only
    return Ezz #*eV_per_hart


####################################################################################################

# POREZAG

####################################################################################################
#@njit
def porezag_hopping(dR,params=None):
    """pairwise Slater Koster hopping parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 7)

    aa = 1.0  # [Bohr radii]
    b = 7.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    if params is None:
        Cpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,
                            -0.0673216, 0.0316900, -0.0117293, 0.0033519, 
                            -0.0004838, -0.0000906])
        Cpp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986, 
                        -0.0300733, 0.0074465, -0.0008563, -0.0004453, 
                        0.0003842, -0.0001855])
    else:
        Cpp_sigma = params[0,:]
        Cpp_pi = params[1,:]
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat*eV_per_hart

def porezag_overlap(dR,params=None):
    """pairwise Slater Koster overlap parameters for pz orbitals of carbon as parameterized by Porezag in
     "Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Overlap matrix elements [eV]
    """
    eV_per_hart=27.2114
    dRn = np.linalg.norm(dR, axis=1)
    #dRn = norm(dR)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    #r = norm(dR)
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 7)
    aa = 1 #; %Angstrom

    b =7 #; %Angstrom

    y = (2*r-(b+aa))/(b-aa)

    
    #overlap matrix coefficient (No units mentioned)
    #originally sigma
    if params is None:
        Cpp_sigma=np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,
                                0.0753818, -0.0108677, -0.0075444, 0.0051533,
                                -0.0013747, 0.0000751])
        Cpp_pi=   np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,
                                0.0061645, 0.0051460, -0.0032776, 0.0009119,
                                -0.0001265, -0.000227])
    else:
        Cpp_sigma = params[0,:]
        Cpp_pi = params[1,:]
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 

    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi
    return Ezz #*eV_per_hart

###########################################################################################

# Moon-Koshino

##########################################################################################

def mk_hopping(descriptors,parameters,grad=False):
    #descriptors = displacements
    #parameters = a,b,c
    ang_per_bohr = 0.529177249 # Ang/bohr radius
    [a,b,c] = parameters #units = [eV, 1/Angstroms, eV]
    x_val = descriptors[:,0]
    y_val = descriptors[:,1]
    z_val = descriptors[:,2]

    a0 = 2.68 * ang_per_bohr
    d0 = 6.33 * ang_per_bohr

    x, y, z = symbols('x y z')
    d = sqrt((x**2 + y **2 + z**2))
    n_sq = z**2 / (x**2 + y **2 + z**2)

    V_p = a * exp(-b * (d - a0))
    V_s = c * exp(-b * (d - d0))
    

    hop_expr = V_p*(1 - n_sq) + V_s * n_sq
    hop_expr = lambdify([x,y,z], hop_expr, "numpy") 
    hop_val = hop_expr(x_val, y_val, z_val)
    if grad:
        hop_diff_x_expr = diff(hop_expr,(x))
        hop_diff_x_expr = lambdify([x,y,z], hop_diff_x_expr, "numpy") 
        hop_diff_y_expr = diff(hop_expr,(y))
        hop_diff_y_expr = lambdify([x,y,z], hop_diff_y_expr, "numpy")
        hop_diff_z_expr = diff(hop_expr,(z))
        hop_diff_z_expr = lambdify([x,y,z], hop_diff_z_expr, "numpy")

        hop_diff_x = hop_diff_x_expr([x_val, y_val, z_val])
        hop_diff_y = hop_diff_y_expr([x_val, y_val, z_val])
        hop_diff_z = hop_diff_z_expr([x_val, y_val, z_val])
        return hop_val, np.stack((hop_diff_x,hop_diff_y,hop_diff_z))
    return hop_val
     

#################################################################################################

# LETB  

################################################################################################

def letb_intralayer(descriptors,parameters,grad=False):
    distances = descriptors[4]
    distances = np.sqrt(distances[0]**2 + distances[1]**2)
    min_distance = min(distances)

    # NN should be within 5% of min_distance
    t01_ix = (distances >= 0.95 * min_distance) & (distances <= 1.05 * min_distance)

    # NNN should be withing 5% of sqrt(3)x of min_distance
    t02_ix = (distances >= 0.95 * np.sqrt(3) * min_distance) & (distances <= 1.05 * np.sqrt(3) * min_distance)

    # NNNN should be within 5% of 2x of min_distance
    t03_ix = (distances >= 0.95 * 2 * min_distance) & (distances <= 1.05 * 2 * min_distance)
   
    # Anything else, we zero out
    t00 = (distances < 0.95 * min_distance) | (distances > 1.05 * 2 * min_distance)

    fit = parameters
    t01 = np.dot(descriptors[0], fit['t01'][1:]) + fit['t01'][0]
    t02 = np.dot(descriptors[1], fit['t02'][1:]) + fit['t02'][0]
    t03 = np.dot(descriptors[2], fit['t03'][1:]) + fit['t03'][0]
    hoppings = np.zeros(len(distances))
    hoppings[t01_ix] = t01
    hoppings[t02_ix] = t02
    hoppings[t03_ix] = t03
    hoppings[t00] = 0
    return hoppings

def letb_interlayer(descriptors,parameters,grad=False):
    
    [a0, b0, c0, a3, b3, c3, a6, b6, c6, d6] = parameters
    r, theta12, theta21 = descriptors[:,0],descriptors[:,1],descriptors[:,2]
    r = r / 4.649 

    v0 = a0 * np.exp(-b0 * r ** 2) * np.cos(c0 * r)
    v3 = a3 * (r ** 2) * np.exp(-b3 * (r - c3) ** 2) 
    v6 =  a6 * np.exp(-b6 * (r - c6)**2) * np.sin(d6 * r)
    hoppings =  v0 
    hoppings += v3 * (np.cos(3 * theta12) + np.cos(3 * theta21))
    hoppings += v6(r, a6, b6, c6, d6) * (np.cos(6 * theta12) + np.cos(6 * theta21))
    
    return hoppings

############################################################################################

# Extras

#############################################################################################


def get_recip_cell(cell):
    """find reciprocal cell given real space cell
    :param cell: (np.ndarray [3,3]) real space cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector
    
    :returns: (np.ndarray [3,3]) reciprocal cell of system where recip_cell[i, j] is the jth Cartesian coordinate of the ith reciprocal cell vector"""
    a1 = cell[:, 0]
    a2 = cell[:, 1]
    a3 = cell[:, 2]

    volume = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return np.array([b1, b2, b3])

def k_uniform_mesh(mesh_size):
    r""" 
    Returns a uniform grid of k-points that can be passed to
    passed to function :func:`pythtb.tb_model.solve_all`.  This
    function is useful for plotting density of states histogram
    and similar.

    Returned uniform grid of k-points always contains the origin.

    :param mesh_size: Number of k-points in the mesh in each
        periodic direction of the model.
        
    :returns:

        * **k_vec** -- Array of k-vectors on the mesh that can be
    """
        
    # get the mesh size and checks for consistency
    use_mesh=np.array(list(map(round,mesh_size)),dtype=int)
    # construct the mesh
    
    # get a mesh
    k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1],0:use_mesh[2]]
    # normalize the mesh
    norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
    norm=norm.reshape(use_mesh.tolist()+[3])
    norm=norm.transpose([3,0,1,2])
    k_vec=k_vec/norm
    # final reshape
    k_vec=k_vec.transpose([1,2,3,0]).reshape([use_mesh[0]*use_mesh[1]*use_mesh[2],3])
    return k_vec

def k_path(sym_pts,nk,report=False):
    r"""

    Interpolates a path in reciprocal space between specified
    k-points.  In 2D or 3D the k-path can consist of several
    straight segments connecting high-symmetry points ("nodes"),
    and the results can be used to plot the bands along this path.
    
    The interpolated path that is returned contains as
    equidistant k-points as possible.

    :param kpts: Array of k-vectors in reciprocal space between
        which interpolated path should be constructed. These
        k-vectors must be given in reduced coordinates.  As a
        special case, in 1D k-space kpts may be a string:

        * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
        * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
        * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)

    :param nk: Total number of k-points to be used in making the plot.
    
    :param report: Optional parameter specifying whether printout
        is desired (default is True).

    :returns:

        * **k_vec** -- Array of (nearly) equidistant interpolated
        k-points. The distance between the points is calculated in
        the Cartesian frame, however coordinates themselves are
        given in dimensionless reduced coordinates!  This is done
        so that this array can be directly passed to function
        :func:`pythtb.tb_model.solve_all`.

        * **k_dist** -- Array giving accumulated k-distance to each
        k-point in the path.  Unlike array *k_vec* this one has
        dimensions! (Units are defined here so that for an
        one-dimensional crystal with lattice constant equal to for
        example *10* the length of the Brillouin zone would equal
        *1/10=0.1*.  In other words factors of :math:`2\pi` are
        absorbed into *k*.) This array can be used to plot path in
        the k-space so that the distances between the k-points in
        the plot are exact.

        * **k_node** -- Array giving accumulated k-distance to each
        node on the path in Cartesian coordinates.  This array is
        typically used to plot nodes (typically special points) on
        the path in k-space.
    """

    k_list=np.array(sym_pts)

    # number of nodes
    n_nodes=k_list.shape[0]

    mesh_step = nk//(n_nodes-1)
    mesh = np.linspace(0,1,mesh_step)
    step = (np.arange(0,mesh_step,1)/mesh_step)

    kvec = np.zeros((0,3))
    knode = np.zeros(n_nodes)
    for i in range(n_nodes-1):
        n1 = k_list[i,:]
        n2 = k_list[i+1,:]
        diffq = np.outer((n2 - n1),  step).T + n1

        dn = np.linalg.norm(n2-n1)
        knode[i+1] = dn + knode[i]
        if i==0:
            kvec = np.vstack((kvec,diffq))
        else:
            kvec = np.vstack((kvec,diffq))
    kvec = np.vstack((kvec,k_list[-1,:]))

    dk_ = np.zeros(np.shape(kvec)[0])
    for i in range(1,np.shape(kvec)[0]):
        dk_[i] = np.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]

    return (kvec,dk_, knode)
