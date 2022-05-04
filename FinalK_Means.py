from mpi4py import MPI
from scipy import misc
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#height = img.shape[0]
#width = img.shape[1]
#misc.imshow(im)

#print(height, width)
clusters = []

def K_mean(k, arr, appr=1):
    if appr == 1:
        arr = np.ndarray()
        clusters = []
        n = len(arr)
        s = n/k
        for i in range(0,n-1,s):
            clusters.append(i)
    maping = dict()

def Clasify(img,clusters):
    height = img.shape[0]
    width = img.shape[1]

    all =[ [] for i in range(len(clusters)) ]
    #all = [[] , [] , []]
    mini = 10
    indx = 0
    for i in range(height):
        for j in range(width):
            mini = 100000000000000000
            for k in range(len(clusters)):
                if np.abs(img[i][j]-clusters[k]) <= mini:
                    indx=k
                    mini = abs(img[i][j]-clusters[k])
                #print('pixl', img[i][j])
                #print('clus', clusters[k])
                #print('min', mini)

            #print('indx',indx)
            #print('all',len(all))
            all[indx].append(img[i][j])
    #print(all)
    return all


K =3
epochs = 3
print(rank)
if rank == 0:
    img = misc.imread('chess.jpg')
    finImage = img
    height = img.shape[0]
    width = img.shape[1]

    clusters= (np.random.randint(0,255,K))
    clusters = np.asarray(clusters)
    print(clusters)
    #clusters.append(0)
    #clusters.append(255)
    #clusters.append(120)
    if size >1:
        chunks = np.array_split(img, size - 1, 0)
        for slave_node_index in range(0, size-1):
            comm.send(chunks[slave_node_index], dest=slave_node_index + 1)
    R_clusters = clusters
    G_clusters = clusters
    B_clusters = clusters
    #-----------------------------------------
    for epo in range(epochs):
        if size >1:
            for slave_node_index in range(0, size - 1):
                comm.send(R_clusters, dest=slave_node_index + 1, tag=10000 + epo)
            for slave_node_index in range(0, size - 1):
                comm.send(G_clusters, dest=slave_node_index + 1,tag = 20000+epo)
            for slave_node_index in range(0, size - 1):
                comm.send(B_clusters, dest=slave_node_index + 1,tag = 30000+epo)

        for i in range(len(clusters)):
            print(clusters[i])

        R_results = []
        G_results = []
        B_results = []
        if size ==1:
            R_results.append(Clasify(img[:,:,0],R_clusters))
            G_results.append(Clasify(img[:, :, 1], G_clusters))
            B_results.append(Clasify(img[:, :, 2], B_clusters))

        if size > 1:
            for slave_node_index in range(0, size-1):
                L = comm.recv(source=slave_node_index + 1,tag=1+epo)
                R_results.append(L[0])
                G_results.append(L[1])
                B_results.append(L[2])

        List_of_all_R = []
        List_of_all_G = []
        List_of_all_B = []

        for i in range(len(clusters)):
            List_of_all_R.append([])
            List_of_all_G.append([])
            List_of_all_B.append([])
            #########
        #updating centroids
        for i in range(len(R_results)):
            for j in range(len(R_results[i])):
                 List_of_all_R[j]+=R_results[i][j]
        for i in range(len(G_results)):
            for j in range(len(G_results[i])):
                 List_of_all_G[j]+=G_results[i][j]
        for i in range(len(B_results)):
            for j in range(len(B_results[i])):
                 List_of_all_B[j]+=B_results[i][j]

        for i in range(len(R_clusters)):
            R_clusters[i] = np.mean(List_of_all_R[i])

        for i in range(len(G_clusters)):
            G_clusters[i] = np.mean(List_of_all_G[i])

        for i in range(len(B_clusters)):
            B_clusters[i] = np.mean(List_of_all_B[i])

        print('epoch',epo,' finished')
    #-------------------------------------------
    print('Red clusters')
    for i in range(len(R_clusters)):
        print(R_clusters[i])
    print('Green clusters')
    for i in range(len(G_clusters)):
        print(G_clusters[i])
    print('Blue clusters')
    for i in range(len(B_clusters)):
        print(B_clusters[i])

    R_indx=0
    G_indx=0
    B_indx=0
    for i in range(height):
        for j in range(width):
            mini = 100000000000000000
            for k in range(len(R_clusters)):
                if np.abs(img[i][j][0] - R_clusters[k]) <= mini:
                    R_indx = k
                    mini = abs(img[i][j][0] - R_clusters[k])
            mini = 100000000000000000
            for k in range(len(G_clusters)):
                if np.abs(img[i][j][1] - G_clusters[k]) <= mini:
                    G_indx = k
                    mini = abs(img[i][j][1] - G_clusters[k])
            mini = 100000000000000000
            for k in range(len(B_clusters)):
                if np.abs(img[i][j][2] - B_clusters[k]) <= mini:
                    B_indx = k
                    mini = abs(img[i][j][2] - B_clusters[k])

            finImage[i][j][0] = R_clusters[R_indx]
            finImage[i][j][1] = G_clusters[G_indx]
            finImage[i][j][2] = B_clusters[B_indx]
    misc.imshow(finImage)
    # print(all)

'''
    imm = img
    for i in range(width):
        for j in rank(height):
            mini = 100000
            for k in range(len(clusters)):
                if (img[i][j] - clusters[k]) < mini:
                    indx = k
            all[indx].append(img[i][j])

'''
#clusters = comm.bcast(clusters, root=0)

if rank != 0:

    chunk_Pic = comm.recv(source=0)
    #----------------------------------------
    for epo in range(epochs):
        R_clusters = comm.recv(source=0,tag=10000+epo)
        G_clusters = comm.recv(source=0, tag=20000 + epo)
        B_clusters = comm.recv(source=0, tag=30000 + epo)

        R_list = Clasify(chunk_Pic[:,:,0], R_clusters)
        G_list = Clasify(chunk_Pic[:,:,1], G_clusters)
        B_list = Clasify(chunk_Pic[:,:,2], B_clusters)
        comm.send([R_list, G_list, B_list], dest=0,tag=1+epo)
        #----------------------------------------
