def laplacian(W, normalized=True):
    """
    Return the Laplacian of the weigth matrix.
    W: input adjacency or weight matrix.
    
    """

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L
 


def coarsen(A, levels, normalized):
    graphs, parents = coarsening.metis(A, levels) #Coarsen a graph multiple times using Graclus variation of the METIS algorithm. 
                                                  #Basically, we randomly sort the nodes, we iterate on them and we decided to group each node
                                                  #with the neighbor having highest w_ij * 1/(\sum_k w_ik) + w_ij * 1/(\sum_k w_kj) 
                                                  #i.e. highest sum of probabilities to randomly walk from i to j and from j to i.
                                                  #We thus favour strong connections (i.e. the ones with high weight wrt all the others for both nodes) 
                                                  #in the choice of the neighbor of each node.
                    
                                                  #Construction is done a priori, so we have one graph for all the samples!
                    
                                                  #graphs = list of spare adjacency matrices (it contains in position 
                                                  #          0 the original graph)
                                                  #parents = list of numpy arrays (every array in position i contains 
                                                  #           the mapping from graph i to graph i+1, i.e. the idx of
                                                  #           node i in the coarsed graph -> that is, the idx of its cluster) 
    perms = coarsening.compute_perm(parents) #Return a list of indices to reorder the adjacency and data matrices so
                                             #that two consecutive nodes correspond to neighbors that should be collapsed
                                             #to produce the coarsed version of the graph.
                                             #Fake nodes are appended for each node which is not grouped with anybody else
    laplacians = []
    for i,A in enumerate(graphs):
        M, M = A.shape

        # We remove self-connections created by metis.
        A = A.tocoo()
        A.setdiag(0)

        if i < levels: #if we have to pool the graph 
            A = coarsening.perm_adjacency(A, perms[i]) #matrix A is here extended with the fakes nodes
                                                       #in order to do an efficient pooling operation
                                                       #in tensorflow as it was a 1D pooling

        A = A.tocsr()
        A.eliminate_zeros()
        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added), |E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

        L = laplacian(A, normalized)
        laplacians.append(L)

    return laplacians, parents, perms[0] if len(perms) > 0 else None

  


  
#Cayley filter definition:
def cayley_operator(L, r, scale, coefficients): # r is the order of the polynomial
    s = np.linalg.norm(L, 2)
    h = 2.0*scale/s
    N = L.shape[0]
    A = (h*L - 1j*np.identity(N)) @ np.linalg.inv(h*L + 1j*np.identity(N))
    AA = np.eye(A.shape[0])
    res = np.eye(A.shape[0])*coefficients[0]
    for k in range(r):
        res = res + 2.0* coefficients[k+1] * (AA @ A).real
        AA = AA @ A
    return res.real

  
  
def poly_operator(L, r, coefficients):
    AA = np.eye(L.shape[0])
    res = np.eye(L.shape[0])*coefficients[0]
    for k in range(r):
        res = res + coefficients[k+1] * (AA @ L)
        AA = AA @ L
    return res

  
  
def interpolate(feature):
    N, M = feature.shape
    new_feature = np.zeros((2*N, M))
    for i in range(N):
        new_feature[2*i, :] = feature[i,:]
        new_feature[2*i + 1, :] = feature[i,:]
    return new_feature

  
  
def pooling(feature, perm, dim_graph):
    feature_after_pool = np.zeros((feature.shape[0]//2, feature.shape[1]))
    for i in range(len(feature)//2):
        if perm[2*i] >=dim_graph and perm[2*i+1] >=dim_graph:
            print('THERE IS ERROR WHEN COARENING')
        if perm[2*i] >=dim_graph:
            feature_after_pool[i, :] = feature[2*i+1,:]
        elif perm[2*i+1] >=dim_graph:
            feature_after_pool[i, :] = feature[2*i,:]     
        else:
            feature_after_pool[i, :] = (feature[2*i,:] + feature[2*i+1,:])/2.0
    return feature_after_pool
  
  

# Sampling operator
def construct_S(perm, m, n): # m is the dimension of the coarsened graph, n is the original dimension
    S = np.zeros((m,n))
    for i in range(m):
      if perm[2*i] >=n and perm[2*i+1] >=n:
        print('THERE IS ERROR WHEN COARENING')
      if perm[2*i] >=n:
        S[i,perm[2*i+1]] = 1.0
      elif perm[2*i+1] >=n:
        S[i,perm[2*i]] = 1.0    
      else:
        S[i,perm[2*i]] = 1.0/np.sqrt(2)
        S[i,perm[2*i+1]] = 1.0/np.sqrt(2)
    return S
              

# Sampling operator with coefficients 0.5
def construct_S_OK(perm, m, n): # m is the dimension of the coarsened graph, n is the original dimension
    S = np.zeros((m,n))
    for i in range(m):
      if perm[2*i] >=n and perm[2*i+1] >=n:
        print('THERE IS ERROR WHEN COARENING')
      if perm[2*i] >=n:
        S[i,perm[2*i+1]] = 1.0
      elif perm[2*i+1] >=n:
        S[i,perm[2*i]] = 1.0    
      else:
        S[i,perm[2*i]] = 1.0/2.0
        S[i,perm[2*i+1]] = 1.0/2.0
    return S
    

    
def convert_map(perm):
    new_perm = np.arange(len(perm))
    new_perm = new_perm[perm]
    new_perm = np.argsort(new_perm)
    return new_perm

  
  
def convert_to_rgb(eigenvector):
    """ Input must be normalized eigenvector """
    
    cmap = matplotlib.cm.get_cmap('coolwarm')
    rgba = cmap(eigenvector) 
    rgba *= 255
    rgba = rgba.astype(int)   
    return rgba
  
  
  
def normalize(chosen_eigen):
    """ Normalize eigenvector before shifting and than dividing by max entry"""

    min_value = chosen_eigen.min(axis=0)
    chosen_eigen_shifted = chosen_eigen - min_value

    max_value = chosen_eigen_shifted.max(axis=0)
    chosen_eigen_shifted_normalized = chosen_eigen_shifted / max_value
    return chosen_eigen_shifted_normalized