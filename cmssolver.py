# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:15:39 2016

@author: jeff
"""

from pyansys import Reader

#from ANSYScdb import CDB_Reader

from scipy.io import loadmat

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from scipy.sparse.linalg import eigsh

import numpy as np


#==============================================================================
# Initialize
#==============================================================================
 
#Kbb=np.zeros((22*mdof.size/2,22*mdof.size/2))
 

#==============================================================================
# Read Sector .full file 
#==============================================================================

#for i in range(2):

for i in range(1,2):
    print i
    fobj = Reader.FullReader('./ANSYS files/Sector%s.full' %i)
    fobj.LoadFullKM()
    
    ndim = fobj.nref.size
    
    # print results
    print('Loaded {:d} x {:d} mass and stiffness matrices'.format(ndim, ndim))
    print('\t k has {:d} entries'.format(fobj.krows.size))
    print('\t m has {:d} entries'.format(fobj.mrows.size))
    
    #==============================================================================
    # Convert fobj data to sparse K & M 
    #==============================================================================
    
    kspmat=csr_matrix((fobj.kdata, (fobj.krows,fobj.kcols)), shape=(ndim,ndim)).toarray()
    
    mspmat=csr_matrix((fobj.mdata, (fobj.mrows,fobj.mcols)), shape=(ndim,ndim)).toarray()
    
    #==============================================================================
    # Remove constrained DOF
    #==============================================================================
    #nncon = np.loadtxt("./ANSYS files/fixed%s.txt" %i, usecols = (0,)) 
    #nncon2 = loadmat('./ANSYS files/FixedBoreNodeSec.mat')
    #nncon2=nncon2['FixedBoreNodeSec'][:,0]
    
    nncon = loadmat('./ANSYS files/FixedBoreNodeRotor.mat')
    
    nncon=nncon['FixedBoreRotor'][:,0]
    
    cdof=np.where(np.in1d(fobj.nref,nncon))[0]
                    
    kspmat=np.delete(kspmat, (cdof), axis=0)
    
    kspmat=np.delete(kspmat, (cdof), axis=1)
    
    mspmat=np.delete(mspmat, (cdof), axis=0)
    
    mspmat=np.delete(mspmat, (cdof), axis=1)
    
    uconst_dof=np.delete(fobj.nref, (cdof))     #do I need this?
            
    fobj.nref=np.delete(fobj.nref, (cdof))      #remove cdof so the mdof are ID'd
                                                #correctly
    
    #vals, vecs = eigsh(kspmat, 4, mspmat, sigma=0, which='LM', tol=1E-1)
    #print np.sqrt(vals)/(2*np.pi)
    #==============================================================================
    #  Read constraint(master) node numbers & partition internal & boundary
    #=============================================    Mcb[x.T, y.T] = mibcb =================================
    """
     Partioning of the substructure matrix into interal (ii) and boundary (bb) dof
    """
    #nnmaster1 = np.loadtxt("./ANSYS files/G1Node%s.txt" %i, usecols = (0,))  # odd way of reading, broken by minus sign in delimiter spot for negative x 
    #nnmaster2 = np.loadtxt("./ANSYS files/G2Node%s.txt" %i, usecols = (0,))
    
    # Read interface node files
    f1 = open("./ANSYS files/G1Node%s.txt" %i, "rw+")
    f2 = open("./ANSYS files/G2Node%s.txt" %i, "rw+")
    content1 = f1.readlines()
    content2 = f2.readlines()
    nummaster = np.shape(content1)[0]
    nnmaster1 = np.zeros(nummaster)
    nnmaster2 = np.zeros(nummaster)
    for line in range(nummaster):
        nnmaster1[line] = int(content1[line][:8])
        nnmaster2[line] = int(content2[line][:8])
    
    
    #nnmaster=np.append(data1,data2) # node numbers master nodes
    
    #remove constrained nodes from master
    idx1 = np.where(np.in1d(nnmaster1,nncon))[0] #finds index where nnmaster has nncon nodes
    idx2 = np.where(np.in1d(nnmaster2,nncon))[0]
    nnmaster1=np.delete(nnmaster1,(idx1))
    nnmaster2=np.delete(nnmaster2,(idx2))
    #nnmaster.searchsorted(nncon)
    mdof1=np.where(np.in1d(fobj.nref,nnmaster1))[0] #finds index where nnmaster has nncon nodes 
    mdof2=np.where(np.in1d(fobj.nref,nnmaster2))[0]
    mdof=np.concatenate((mdof1,mdof2))  
     
    nnii = fobj.nref    # make iidof all the dof
    nnii = np.delete(nnii, (mdof)) # subtract the master dof
    nnii = np.unique(nnii)
    
    # Error Check
    #(nnii.size+nnmaster1.size+nnmaster2.size+nncon.size)*3 == ndim   # error check internal dof + master dof...
                                                     # + constraint dof should equal the imported K matrix size 
    
    nnall = fobj.nref[::3] # grabs every third element of the array # could also use np.unique
    
    iidof = np.where(np.in1d(fobj.nref, nnii))[0]  
      
    #now cdof, mdof, and iidof are all computed, partition matrix w/mdof,idof
       
    partidx=np.concatenate((mdof,iidof))   # add master and interal dof indexes
       
    Kp=kspmat[partidx,:]
    Kp=Kp[:,partidx] 
    
    Mp=mspmat[partidx,:]
    Mp=Mp[:,partidx] 
    
    partnref=fobj.nref[partidx] #new reference for node numbers in partitioned K & M
    
    # Remove unnecessary data
    del kspmat, mspmat
    
    #==============================================================================
    # 
    #==============================================================================
    
    ##Error Check
    ##Check to see if the partitioned matrix still gives the eigvals of non partitioned
    vals, vecs = eigsh(Kp, 4, Mp, sigma=0, which='LM', tol=1E-1)
    print np.sqrt(vals)/(2*np.pi)
    
    
    #==============================================================================
    # 
    #==============================================================================
    # SubPartition out ii, bb, bi, ib
    iidofp=np.where(np.in1d(partnref, nnii))[0]
    
    Kpii = Kp[iidofp,:]
    Kpii = Kpii[:,iidofp]
    
    # Partition out bb
    bbdofp1=np.where(np.in1d(partnref,nnmaster1))[0]
    bbdofp2=np.where(np.in1d(partnref,nnmaster2))[0]
    bbdofp=np.concatenate((bbdofp1,bbdofp2))
    
    Kpbb = Kp[bbdofp,:]
    Kpbb = Kpbb[:,bbdofp]
    
    # Partition out ib
    
    Kpib = Kp[iidofp,:]
    Kpib = Kpib[:,bbdofp]
    
    Kpbi = Kpib.T
    
    # Now Mass
    Mpii = Mp[iidofp,:]
    Mpii = Mpii[:,iidofp]
    
    Mpbb = Mp[bbdofp,:]
    Mpbb = Mpbb[:,bbdofp]
    
    Mpib = Mp[iidofp,:]
    Mpib = Mpib[:,bbdofp]
    
    Mpbi = Mpib.T
    
    del Kp, Mp
    
    #==============================================================================
    # Calculate constraint modes
    #==============================================================================
    #
    #phiC = -1 * Kpii\Kpib
    
    phiC = -1 * np.linalg.solve(Kpii, Kpib) # solves Kpii(phiC) = Kpib
    
    #a = np.linalg.inv(Kpii)
    #b= np.dot(a,Kpib)   
    #===================================== =========================================
    # Calculate Constrained Normal Modes Perform Eigensolution
    #==============================================================================
    eignum = 4
    
    vals, phiN = eigsh(Kpii, eignum, Mpii, sigma=0, which='LM', tol=1E-1)
    
    print np.sqrt(vals)/(2*np.pi)
    
    #==============================================================================
    #  Craig Bampton Transformation
    #==============================================================================
    
    kbbcb = Kpbb + np.dot(Kpbi,phiC)
    
    kiicb = np.dot(np.dot(phiN.T,Kpii),phiN) # same as phiN.T * Kpii * phiN
    
    # diagonal of kiicb equals vals
    
    mbbcb = Mpbb + np.dot(phiC.T,Mpib) +np.dot(Mpbi,phiC) + np.dot(np.dot(phiC.T,Mpii),phiC)
    
    mbicb = np.dot(Mpbi,phiN) + np.dot(np.dot(phiC.T,Mpii),phiN)
    
    mibcb = mbicb.T
    
    miicb = np.dot(np.dot(phiN.T,Mpii),phiN)
    
    
    #==============================================================================
    # Couple Substructures
    #==============================================================================
    
    """
    bbdofp  # are the boundary (master) dof locations in the substucture Kp,Mp
            # should always be the first master 0:nnmaster * 3
    partnref # dof node numbers for partitioned matrix
    kbbcb #
    
    """
    if i == 1:
        nmodes = 4
        nsubtotal=22
        totcbdof = nsubtotal*(nmodes+mdof.size/2)    
        Kcb = np.zeros((totcbdof,totcbdof))
        Mcb = np.zeros((totcbdof,totcbdof)) 
        Kbbnref=[]
        Kbb=np.zeros((22*mdof.size/2,22*mdof.size/2))
        Mbb=np.zeros((22*mdof.size/2,22*mdof.size/2))
    
    nsub=i    
    
    kbbcbnref = partnref[bbdofp]
    
    duplidof = np.where(np.in1d(kbbcbnref, Kbbnref))[0]
    
    kbbcbnref = np.delete(kbbcbnref,(duplidof))
    
    Kbbnref.extend(kbbcbnref)
    
    kbbcbnref = partnref[bbdofp]  #setting back to all kbbcb dof for adding to Kbb 
    
    subdof = np.where(np.in1d(Kbbnref,kbbcbnref))[0]
    
    x, y = np.meshgrid(subdof,subdof)
    
    Kbb[x, y] = Kbb[x, y] + kbbcb
    
    
    Mbb[x, y] = Mbb[x, y] + mbbcb
        

    Mibrowstart = (nsub-1)*nmodes
    Mibrowend = (nsub-1)*nmodes+nmodes
    
    Mibcolstart = nsubtotal*nmodes+((nsub-1)*mibcb.shape[1])
    Mibcolend = Mibcolstart+mibcb.shape[1]
        
    KMiirowstart = (nsub-1)*nmodes
    KMiirowend = (nsub-1)*nmodes+nmodes

    KMiicolstart = (nsub-1)*nmodes
    KMiicolend = (nsub-1)*nmodes+nmodes
    
    
    print Mibrowstart
    print Mibrowend
    print Mibcolstart
    print Mibcolend
    
    print KMiirowstart
    print KMiirowend
    
    print KMiicolstart
    print KMiicolend
    

    Kcb[range(KMiirowstart,KMiirowend), range(KMiicolstart,KMiicolend)]=vals
    
    Mcb[range(KMiirowstart,KMiirowend), range(KMiicolstart,KMiicolend)]=vals.shape
    
    x, y =np.meshgrid(range(Mibrowstart,Mibrowend), range(Mibcolstart,Mibcolend))
    
    Mcb[x.T, y.T] = mibcb  # had to transpose x, y to get right locations

    x, y =np.meshgrid(range(Mibcolstart,Mibcolend) ,range(Mibrowstart,Mibrowend))

    Mcb[x, y] = mibcb
    
x, y =np.meshgrid(range(nmodes*nsubtotal,totcbdof), range(nmodes*nsubtotal,totcbdof))    
    
Kcb[x,y]=Kbb

Mcb[x,y]=Mbb

del Kbb, Mbb


"""

146 master nodes per interface

438 master dof per interface 

438 * 22 is kbbsize =9636

How big is Mib, which is aligned with  mbb, only 876 x nnnmodes
"""

