import pytrec_eval
import ipdb
from sparsembed import  utils
from tabulate import tabulate
import json
import argparse
import pandas as pd

def read_the_json(path):
    with open(path) as fw:
        anno = json.loads(fw.read())
    return anno

def read_bm25_dataset(name):
    bm25_anno_ids = []
    bm25_anno =  read_the_json(f'models/{name}_bm25_result.json')
    return bm25_anno

def read_dense_dataset(name, dense):
    dense_anno =  read_the_json(f'models/{dense}_{name}_result.json')
    return dense_anno

def tm2c2(fsem, flex, alpha):
    newres = {}
    for qid in fsem:
        newres[qid] = {}
        res_sem = fsem[qid]
        res_lex = flex[qid]
        scores = []
        for k in res_sem:
            scores.append(res_sem[k])
        sup_sem = max(scores) 
        scores = []
        for k in res_lex:
            scores.append(res_lex[k])
        if len(scores) == 0:
            sup_lex = 0    
        else:
            sup_lex = max(scores) 
        for k in res_sem: 
            newres[qid][k] = alpha * (res_sem[k]-(-1))/(sup_sem- (-1))
        for k in res_lex:
            if sup_lex == 0:
                break
            if k not in newres[qid]:
                newres[qid][k] = 0
            try:
                newres[qid][k] = newres[qid][k] + (1-alpha)*(res_lex[k] - 0)/(sup_lex-0)
            except Exception as e:
                ipdb.set_trace()
    return newres
        
def rrf(fsem, flex, ita):
    newres = {}
    for qid in fsem:
        newres[qid] = {}
        res_sem = fsem[qid]
        res_lex = flex[qid]
        res_sem = dict(sorted(res_sem.items(), key=lambda item: item[1],reverse=True))
        for i, k in enumerate(res_sem):
            res_sem[k] = i 
        res_lex = dict(sorted(res_lex.items(), key=lambda item: item[1],reverse=True))
        for i, k in enumerate(res_lex):
            res_lex[k] = i 
        for k in res_sem:
            newres[qid][k] = 1 / (ita+res_sem[k])
        for k in res_lex: 
            if k not in newres[qid]:
                newres[qid][k] = 0
            newres[qid][k] = newres[qid][k] + 1/ (ita + res_lex[k])
    return newres

def main():
    parser = argparse.ArgumentParser(description='Example script with argparse')
    # Define command-line arguments
    parser.add_argument('--ita', type=int, help='Description for ita argument')
    parser.add_argument('--alpha', type=float, help='Description for alpha argument')
    parser.add_argument('--dense', type=str, help='Description for dense argument')
    parser.add_argument('--topk', type=int, help='Description for topk argument')

    args = parser.parse_args()
    ita = args.ita
    alpha = args.alpha
    dense_name = args.dense
    topk = args.topk

    table = {} 
    table['dataset'] = []
    table[f'sem R@{topk}'] = []
    table[f'lex R@{topk}'] = []
    table[f'tm2c2 R@{topk}'] = []
    table[f'rrf R@{topk}'] = []
    table[f'sem NDCG@{topk}'] = []
    table[f'lex NDCG@{topk}'] = []
    table[f'tm2c2 NDCG@{topk}'] = []
    table[f'rrf NDCG@{topk}'] = []
    
    names = ['hotpotqa', 'fiqa', 'nfcorpus','scifact']
    for name in names:
        table['dataset'].append(name)
        documents, queries, qrels = utils.load_beir(name, split="test")
        
        run_sem = read_dense_dataset(name, dense_name)
        run_lex = read_bm25_dataset(name)
        
        ##for sem evaluatioin
        run_sem = tm2c2(run_sem, run_lex, 1.0) 

        ##for lex evaluation
        run_lex = tm2c2(run_sem, run_lex, 0.0) 

        ##for convex  
        run_tm2c2 = tm2c2(run_sem, run_lex, alpha) 

        ##for rrf 
        run_rrf = rrf(run_sem, run_lex, ita) 


        for it in [('sem',run_sem), ('lex', run_lex), ('tm2c2', run_tm2c2), ('rrf', run_rrf)]:
            run = it[1]
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrels, {'map_cut', 'ndcg_cut',f'recall_{topk}', f'ndcg_cut_{topk}'})
            results = evaluator.evaluate(run)
            scores = []
            ndcg_scores = []
            for result in results:
                scores.append(results[result][f'recall_{topk}'])
                ndcg_scores.append(results[result][f'ndcg_cut_{topk}'])
            recall = sum(scores)/len(scores)
            ndcg = sum(ndcg_scores)/len(ndcg_scores)

            table[f'{it[0]} R@{topk}'].append(recall)
            table[f'{it[0]} NDCG@{topk}'].append(ndcg)
    df = pd.DataFrame(table) 
    markdown_table = tabulate(df, tablefmt='pipe', headers='keys')
    with open(f'results/{ita}_{alpha}_{dense_name}_{topk}.md', 'w') as file:
        file.write(markdown_table)

    
if __name__ == '__main__':
    main()
