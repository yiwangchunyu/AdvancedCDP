import heapq
import time
from math import exp

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from tqdm import tqdm

EPS=40 #邻域半径
P=2

MAX_DELTA=1e10
DISPLAY_SCATTER_SIZE=150
DIS_INF=1e10

class Point():

    def __init__(self,id,x):
        self.id=id
        self.x=x
        self.label_real=-1
        self.label=-1
        self.cost = DIS_INF  # 距离源S的距离
        self.edge_all = {}  # 边的情况
        self.edge_eps = {}  # 边的情况
        self.density = 0
        self.delta = MAX_DELTA
        self.gamma = None
        self.pre=None
        self.pre_size=0
        self.cost_sum=0


    def comput_euclidean_dis_with(self,point_y):
            norm_2 = np.linalg.norm(np.array(self.x) - np.array(point_y.x), ord=2)
            return norm_2

    def __lt__(self, other):
        return self.cost < other.cost


def scatter2D(pts,title=''):
    x=[p.x[0] for  p in pts]
    y=[p.x[1] for  p in pts]
    labels=[p.label for p in pts]
    plt.title(title)
    plt.scatter(x, y, s=10, c=labels, marker='o')
    plt.savefig('scatter2D.png', dpi=500)
    plt.show()


def dataset(data_file, label_file, display:bool=True):
    id2point={}
    pts=[]
    dataf=open(data_file,'r')
    labelf = open(label_file, 'r')
    for line in dataf:
        if line[:2]=='//':
            continue
        id=line.split('\t')[0]
        pos=tuple(map(float,line.split('\t')[1:]))
        point=Point(id,pos)
        pts.append(point)
        id2point[id]=point

    for line in labelf:
        if line[:2]=='//':
            continue
        id = line.split('\t')[0]
        label=int(line.split('\t')[-1])
        id2point[id].label_real=label

    if display:
        x = [p.x[0] for p in pts]
        y = [p.x[1] for p in pts]
        labels = [p.label_real for p in pts]
        plt.title('')
        plt.scatter(x, y, s=DISPLAY_SCATTER_SIZE, c=labels, marker='o')
        plt.axis('scaled')
        plt.savefig('scatter2D.png', dpi=500)
        plt.show()
    return id2point,pts

def find_edge(pts:List[Point]):
    print('find edge...')
    for point in tqdm(pts):
        for neighbor_point in pts:
            if point.id==neighbor_point.id:
                continue
            dis=point.comput_euclidean_dis_with(neighbor_point)
            point.edge_all[neighbor_point.id]=dis
            if dis<EPS:
                point.edge_eps[neighbor_point.id]=dis

def decision_graph(pts:List[Point], display:bool=True) -> List[Point]:
    #计算每个点的密度
    print('computing density...')
    for point in tqdm(pts):
        for id in point.edge_eps:
            point.density+=exp(-(point.edge_eps[id]/EPS)**2)
    #按密度从高到低排序
    pts.sort(key=lambda p:p.density,reverse=True)
    #计算delta
    max_density=pts[0].density
    print('computing delta...')
    for id,point in tqdm(enumerate(pts)):
        if point.density==max_density:
            point.delta = -MAX_DELTA
            for point_neighbor in pts:
                if point.id==point_neighbor.id:
                    continue
                dis = point.edge_all[point_neighbor.id]
                point.delta=max(point.delta,dis)
        else:
            for point_neighbor in pts[:id]:
                if point_neighbor.density>point.density:
                    dis = point.edge_all[point_neighbor.id]
                    point.delta=min(point.delta,dis)
    #delta归一化(max_min nomalization)
    print('nomalization...')
    max_delta=-MAX_DELTA
    min_delta=MAX_DELTA
    for point in pts:
        max_delta=max(max_delta,point.delta)
        min_delta = min(min_delta, point.delta)
    for point in pts:
        point.delta=(point.delta-min_delta)/(max_delta-min_delta)

    if display:
        dens = [p.density for p in pts]
        deltas = [p.delta for p in pts]
        plt.title('decision graph')
        plt.xlabel('density')
        plt.ylabel('delta')
        plt.scatter(dens, deltas, s=20, marker='o')
        plt.show()
    return pts

def find_albow(pts:List[Point], display:bool=True):
    for point in pts:
        point.gamma=point.delta*point.density
    #排序
    pts_sorted_by_gamma=sorted(pts, key=lambda point:point.gamma, reverse=True)
    gammas=[point.gamma for point in pts_sorted_by_gamma]
    # derivative1st = [gammas[i] - gammas[i + 1] for i in range(0, len(gammas)-1)]
    from kneed import KneeLocator
    kn = KneeLocator([i for i in range(len(gammas))], gammas, curve='convex', direction='decreasing')
    if display:
        plt.title('gamma elbow point')
        plt.xlabel('n')
        plt.ylabel('gamma')
        plt.vlines(x=kn.elbow, ymin=0,ymax=gammas[0],colors='r',linestyles = "--", label='elbow=%d'%(kn.elbow))
        plt.scatter([i for i in range(len(gammas))],gammas,s=20,marker='o')
        plt.scatter(kn.elbow, gammas[kn.elbow], s=30, marker='o',c='r')
        plt.legend()
        # plt.scatter([i+1 for i in range(len(derivative1st))], derivative1st, s=20, marker='o',c='r')
        plt.show()

    num_class=kn.elbow
    n_centre_point=pts_sorted_by_gamma[:num_class]
    n_centre_id = [point.id for point in n_centre_point]
    return n_centre_id

def clustring_CDP(n_centre_id:List[int], pts:List[Point], display:bool=True):
    label=1
    for centre_id in n_centre_id:
        id2point[centre_id].label=label
        label+=1

    for point in pts:
        min_dis=MAX_DELTA
        l=-1
        for centre_id in n_centre_id:
            if centre_id==point.id:
                l=id2point[centre_id].label
                break
            dis=point.edge_all[centre_id]
            if dis<min_dis:
                min_dis=dis
                l=id2point[centre_id].label
        point.label=l

    if display:
        x = [p.x[0] for p in pts]
        y = [p.x[1] for p in pts]
        labels = [p.label for p in pts]
        cx = [id2point[id].x[0] for id in n_centre_id]
        cy = [id2point[id].x[1] for id in n_centre_id]
        clabels = [id2point[id].label for id in n_centre_id]
        plt.title('clustring result')
        plt.scatter(x, y, s=DISPLAY_SCATTER_SIZE, c=labels, marker='o')
        plt.scatter(cx, cy, s=DISPLAY_SCATTER_SIZE*3, c=clabels, marker='o')
        plt.axis('scaled')
        plt.savefig('clustring_CDP.png',dpi=500)
        plt.show()


def clustring_DiegoGeneric(n_centre_id:List[Point], pts:List[Point], display:bool=True,arrow:bool=False):
    pq = []  # 优先队列
    label = 1
    inqueue=set()
    solved=set()
    for centre_id in n_centre_id:
        id2point[centre_id].label = label
        label += 1

    for centre_id in n_centre_id:
        id2point[centre_id].cost=0
        heapq.heappush(pq, id2point[centre_id])
        inqueue.add(centre_id)

    while pq:
        top = heapq.heappop(pq)
        inqueue.remove(top.id)
        solved.add(top.id)
        for id in top.edge_all:
            if id in solved:
                continue
            cost_sum=top.cost+pow(top.edge_all[id],P)
            cost=pow(cost_sum,1/P)
            if cost<id2point[id].cost:
                id2point[id].cost=cost
                id2point[id].cost_sum = cost_sum
                id2point[id].pre=top.id
                id2point[id].label=top.label
                if (id not in inqueue) and (id not in solved):
                    heapq.heappush(pq,id2point[id])
                    inqueue.add(id)

    if display:
        if arrow:
            for point in pts:
                if point.pre is not None:
                    plt.arrow(point.x[0],
                              point.x[1],
                              id2point[point.pre].x[0]-point.x[0]-(id2point[point.pre].x[0]-point.x[0])/5,
                              id2point[point.pre].x[1]-point.x[1]-(id2point[point.pre].x[1]-point.x[1])/5,
                              head_width=5,
                              head_length=5,
                              color='red',
                              alpha=0.4)
        x = [p.x[0] for p in pts]
        y = [p.x[1] for p in pts]
        labels = [p.label for p in pts]
        cx = [id2point[id].x[0] for id in n_centre_id]
        cy = [id2point[id].x[1] for id in n_centre_id]
        clabels = [id2point[id].label for id in n_centre_id]
        plt.title('clustring result')
        plt.scatter(x, y, s=DISPLAY_SCATTER_SIZE, c=labels, marker='o')
        plt.scatter(cx, cy, s=DISPLAY_SCATTER_SIZE*3, c=clabels, marker='o')
        plt.axis('scaled')
        plt.savefig('clustring_ours_result.png', dpi=500)
        plt.show()

def clustring_ours(n_centre_id:List[Point], pts:List[Point], display:bool=True,arrow:bool=False):
    pq = []  # 优先队列
    label = 1
    inqueue=set()
    solved=set()
    for centre_id in n_centre_id:
        id2point[centre_id].label = label
        label += 1

    for centre_id in n_centre_id:
        id2point[centre_id].cost=0
        heapq.heappush(pq, id2point[centre_id])
        inqueue.add(centre_id)

    while pq:
        top = heapq.heappop(pq)
        inqueue.remove(top.id)
        solved.add(top.id)
        for id in top.edge_eps:
            if id in solved:
                continue
            cost_sum=top.cost+pow(top.edge_eps[id],P)
            cost=pow(cost_sum,1/P)
            if cost<id2point[id].cost:
                id2point[id].cost=cost
                id2point[id].cost_sum = cost_sum
                id2point[id].pre=top.id
                id2point[id].label=top.label
                if (id not in inqueue) and (id not in solved):
                    heapq.heappush(pq,id2point[id])
                    inqueue.add(id)

    if display:
        if arrow:
            for point in pts:
                if point.pre is not None:
                    plt.arrow(point.x[0],
                              point.x[1],
                              id2point[point.pre].x[0]-point.x[0]-(id2point[point.pre].x[0]-point.x[0])/5,
                              id2point[point.pre].x[1]-point.x[1]-(id2point[point.pre].x[1]-point.x[1])/5,
                              head_width=5,
                              head_length=5,
                              color='red',
                              alpha=0.4)
        x = [p.x[0] for p in pts]
        y = [p.x[1] for p in pts]
        labels = [p.label for p in pts]
        cx = [id2point[id].x[0] for id in n_centre_id]
        cy = [id2point[id].x[1] for id in n_centre_id]
        clabels = [id2point[id].label for id in n_centre_id]
        plt.title('clustring result')
        plt.scatter(x, y, s=DISPLAY_SCATTER_SIZE, c=labels, marker='o')
        plt.scatter(cx, cy, s=DISPLAY_SCATTER_SIZE*3, c=clabels, marker='o')
        plt.axis('scaled')
        plt.savefig('clustring_DiegoGeneric_result.png', dpi=500)
        plt.show()


if __name__=="__main__":
    # id2point,pts=dataset('data/207_ecg_20190312_rr1_rr.txt','data/207_ecg_20190312_rr1_rr.gs.txt')
    id2point,pts=dataset('data/99_synthetic_dendrites.txt','data/99_synthetic_dendrites.gs.txt')
    find_edge(pts)
    decision_graph(pts)
    n_centre_id = find_albow(pts)

    # clustring_CDP(n_centre_id, pts)

    # begin=time.time()
    # clustring_DiegoGeneric(n_centre_id,pts,arrow=True)
    # end=time.time()
    # clustring_DiegoGeneric_duration=end-begin

    begin = time.time()
    clustring_ours(n_centre_id, pts, arrow=True)
    end = time.time()
    clustring_ours_duration = end - begin
    print(clustring_ours_duration)
