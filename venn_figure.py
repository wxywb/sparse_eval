import matplotlib.pyplot as plt
import numpy as np
import json
from sparsembed import  utils
import ipdb
import matplotlib
import  random
import argparse


import matplotlib.pyplot as plt
from matplotlib_venn import venn3,venn3_circles

def read_the_json(path):
    with open(path) as fw:
        anno = json.loads(fw.read())
    return anno

parser = argparse.ArgumentParser(description='Example script with argparse')
parser.add_argument('--dataset', type=str, help='dataset name')
args = parser.parse_args()

name = args.dataset
names = [name]
topk = 10

documents, queries, qrels = utils.load_beir(name, split="test")

only_a   = 0
only_b   = 0
only_a_b = 0 
only_c   = 0 
only_a_c = 0 
only_b_c = 0 
a_b_c    = 0

gt_only_a   = 0 
gt_only_b   = 0
gt_only_c   = 0
gt_only_a_b = 0
gt_only_a_c = 0
gt_only_b_c = 0
gt_a_b_c    = 0

missed_gt = 0
total_gt  = 0

#fig, ax = plt.subplots()

list_a = []
list_b = []
list_c = []
list_gt = []

cnt = 0
for name in names:
    splade_anno = read_the_json(f'./output/splade_{name}.json')
    bge_anno =  read_the_json(f'./output/prompt_bge_{name}_result.json')
    bm25_anno =  read_the_json(f'./output/{name}_bm25_result.json')

    #print(splade_anno)
    i = 0 
    for qid in splade_anno:
        print( i, qid)
        i = i + 1
        #ipdb.set_trace()
        gt = qrels[qid]
        splade_anno_ids = splade_anno[qid]
        bge_anno_ids = []  
        bm25_anno_ids = []
        for k in bge_anno[qid]:
            bge_anno_ids.append({'id': k , 'similarity': bge_anno[qid][k]})
        bge_anno_ids = sorted(bge_anno_ids, key=lambda x: x['similarity'], reverse=True)
        for k in bm25_anno[qid]:
            bm25_anno_ids.append({'id': k , 'similarity': bm25_anno[qid][k]})
        bm25_anno_ids = sorted(bm25_anno_ids, key=lambda x: x['similarity'], reverse=True)
          
        ids_a =  [item['id'] for item in splade_anno_ids[:topk]]
        ids_b =  [item['id'] for item in bge_anno_ids[:topk]]
        ids_c =  [item['id'] for item in bm25_anno_ids[:topk]]
        
        list_a.extend(ids_a)
        list_b.extend(ids_b)
        list_c.extend(ids_c)
        list_gt.extend(gt)
        set_a = set(ids_a[:topk]) 
        set_b = set(ids_b[:topk]) 
        set_c = set(ids_c[:topk]) 
        #if len(set_c) != topk:
            #ipdb.set_trace()
                    
        for _id in ids_b:
            if _id in gt:
                cnt += 1
        ids_a_n_b_n_c = set_a.intersection(set_b).intersection(set_c)   
        ids_a_n_b = set_a.intersection(set_b) - set_c
        ids_a_n_c = set_a.intersection(set_c) - set_b
        ids_c_n_b = set_c.intersection(set_b) - set_a 
        
        exclusive_a = set_a - set_b - set_c
        exclusive_b = set_b - set_a - set_c
        exclusive_c = set_c - set_a - set_b

        #ipdb.set_trace()
        #global only_a
        #global only_a, only_b, only_c, only_a_b, only_a_c, only_b_c, a_b_c
        only_a = only_a + len(exclusive_a)
        print(only_a)
        only_b = only_b + len(exclusive_b)
        only_c = only_c + len(exclusive_c)

        only_a_b = only_a_b + len(ids_a_n_b)
        only_a_c = only_a_c + len(ids_a_n_c)
        only_b_c = only_b_c + len(ids_c_n_b)

        a_b_c = a_b_c + len(ids_a_n_b_n_c)

        gt_only_a = gt_only_a + len(exclusive_a.intersection(gt))
        gt_only_b = gt_only_b + len(exclusive_b.intersection(gt))
        gt_only_c = gt_only_c + len(exclusive_c.intersection(gt))
        gt_only_a_b = gt_only_a_b + len(ids_a_n_b.intersection(gt))
        gt_only_a_c = gt_only_a_c + len(ids_a_n_c.intersection(gt))
        gt_only_b_c = gt_only_b_c + len(ids_c_n_b.intersection(gt))
        gt_a_b_c = gt_a_b_c + len(ids_a_n_b_n_c.intersection(gt))

        #assert(gt_only_a + gt_only_b+ gt_only_c+ gt_only_a_b + gt_only_a_c + gt_only_b_c+gt_a_b_c  )
    
        total_gt = total_gt + len(gt)
        miss = len(set(gt) - set_a - set_b - set_c )
        missed_gt = missed_gt + miss
        if gt_only_a + gt_only_b+ gt_only_c+ gt_only_a_b + gt_only_a_c + gt_only_b_c+gt_a_b_c+missed_gt != total_gt:
            ipdb.set_trace()
        rend_points = []
        in_range  = [0,0.25]
        out_range  = [0.25,0.5]

        for id_ in ids_a_n_b_n_c:
            rend_points.append({'id':id_,'color':'wo', 'range': [0,12], 'radius':in_range})
        for id_ in exclusive_a:
            rend_points.append({'id':id_,'color':'ro','range': [0,4], 'radius':out_range})
        for id_ in exclusive_b:
            rend_points.append({'id':id_,'color':'bo','range': [4,8], 'radius':out_range})
        for id_ in exclusive_c:
            rend_points.append({'id':id_,'color':'go','range': [8,12], 'radius':out_range})
        for id_ in ids_a_n_b:
            rend_points.append({'id':id_,'color':'mo','range': [3.5,4.5], 'radius':out_range})
        for id_ in ids_c_n_b:
            rend_points.append({'id':id_,'color':'co','range': [7.5,8.5], 'radius':out_range})
        for id_ in ids_a_n_c:
            rend_points.append({'id':id_,'color':'yo','range': [11.5,12.5], 'radius':out_range})

        for point_ in rend_points:
            if point_['id'] in gt:
                point_['is_gt'] = True
            else:
                point_['is_gt'] = False

            #save_points_on_clock(ax, point_)


def venn_diagram(a, b, c, labels=['splade', 'bge', 'bm25']):

    global only_a, only_b, only_c, only_a_b, only_a_c, only_b_c, a_b_c
    global gt_only_a, gt_only_b, gt_only_c, gt_only_a_b, gt_only_a_c, gt_only_b_c, gt_a_b_c
    global missed_gt
    global total_gt
    global name
    gts = [gt_only_a, gt_only_b, gt_only_c, gt_only_a_b, gt_only_a_c, gt_only_b_c, gt_a_b_c]

    #ipdb.set_trace()
    def temp(s):
        gts_kv = {}
        gts_kv[only_a] = gt_only_a 
        gts_kv[only_b] = gt_only_b 
        gts_kv[only_c] = gt_only_c 
        gts_kv[only_a_b] = gt_only_a_b 
        gts_kv[only_b_c] = gt_only_b_c
        gts_kv[only_a_c] = gt_only_a_c 
        gts_kv[a_b_c] = gt_a_b_c

        return f'{s}({gts_kv[s]})[{str(gts_kv[s]/total_gt* 100)[:4]}%]'

    a_part = only_a+only_a_b+only_a_c + a_b_c
    b_part = only_b+only_a_b+only_b_c + a_b_c
    c_part = only_c+only_a_c+only_b_c + a_b_c

    a_part_gt = gt_only_a+gt_only_a_b+gt_only_a_c + gt_a_b_c
    b_part_gt = gt_only_b+gt_only_a_b+gt_only_b_c + gt_a_b_c
    c_part_gt = gt_only_c+gt_only_a_c+gt_only_b_c + gt_a_b_c

     
    labels=[f'splade\n {a_part}({a_part_gt})[{str(a_part_gt/total_gt * 100)[:4]}%]', f'bge\n {b_part}({b_part_gt})[{str(b_part_gt/total_gt * 100)[:4]}%]', f'bm25\n {c_part}({c_part_gt})[{str(c_part_gt/total_gt * 100)[:4]}%]']    
    venn_figure = venn3(subsets=(only_a, only_b, only_a_b, only_c, only_a_c, only_b_c, a_b_c), set_labels=labels, subset_label_formatter=temp)
    for patch,gt in zip(venn_figure.patches, gts):
        vs = venn_figure.patches[0].get_path().vertices
        center_x =  vs[:, 0].mean()
        center_y =  vs[:, 1].mean()

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.gca().invert_yaxis()
plt.clf()
#fig, ax = plt.subplots()
venn_diagram(list_a, list_b, list_c)

ax.text(0.5, -0.5, f'{name} missed gt: {missed_gt} {str(missed_gt/total_gt* 100)[:4]}%]', va='center', ha='center', color='black')

plt.savefig(f'./figures/figure_{name}.png')


print('cnt',cnt)
