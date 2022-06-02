from typing import List
import pandas as pd
EPS=1e-12

class Eval():
    """
    Instantiate the Eval class with path to test file.

    Example Code:
    import Evaluate
    evaluator = Evaluate.Eval('path/to/test_qrel')
    evaluator.evaluate(q_id_list, predict_ids_list)
    """

    def __init__(self, qrels_file_path: str):
        self.qrels = pd.read_csv(qrels_file_path, sep="\t", header=None, names=['id_left', '0', 'id_right', 'label'])

    def _hits_at_k(self, query_id: int, predict_ids: List[int], k: int):
        """
        ARGS:
        query_id (int): id_left from queries.csv
        predict_ids (int list): list of generated predictions, sorted by confidence, with index 0 = highest confidence
        k (int): top-k predictions to look at 
        
        RETURNS:
        ret (bool): if query id is in top k of predict_ids, then return 1, else return 0 for that particular query 
        """
        golden = self.qrels[(self.qrels['id_left']==query_id) & (self.qrels['label']==2)].iloc[0]['id_right'] # get id of article corresponding to query
        top_k_list = predict_ids[:k]
        if golden in top_k_list:
            ret = 1
        else:
            ret = 0
        return ret

    def _avg_hits_at_k(self, q_id_list: List[int], predict_ids_list: List[List[int]], k: int):
        """
        ARGS:
        q_id_list (int list): consolidated query ids (id_left) from queries.csv
        predict_ids_list (list (int list)): consolidated list of generated predictions
        k (int): top-k predictions to look at 

        RETURNS:
        avg_hak (float): Averaged Hits @ k
        """
        no_of_queries = 0
        cum_hak = 0
        for index, query in enumerate(q_id_list):
            if  len(predict_ids_list[index]) < k:
                raise ValueError('k is greater than number of predicted_id!')
            no_of_queries+=1
            hak = self._hits_at_k(query, predict_ids_list[index], k)
            cum_hak+=hak
        avg_hak = cum_hak/(no_of_queries+EPS)
        return avg_hak

    def _reciprocal_rank(self, query_id: int , predict_ids: List[int]):
        """
        ARGS:
        query_id (int): id_left from queries.csv
        predict_ids (int list): list of generated predictions

        RETURNS:
        rr (float): Reciprocal rank of a particular query
        """
        golden = self.qrels[(self.qrels['id_left']==query_id) & (self.qrels['label']==2)].iloc[0]['id_right'] # get id of article corresponding to query
        if not golden in predict_ids:
            rr = 0
        else:
            ind = predict_ids.index(golden)+1 # Plus 1 because index starts from 0!!! 
            rr = 1/(ind)
        return rr


    def _mean_reciprocal_rank(self, q_id_list: List[int], predict_ids_list: List[List[int]]):
        """
        ARGS:
        q_id_list (int list): consolidated query ids (id_left) from queries.csv
        predict_ids_list (list (int list)): consolidated list of generated predictions

        RETURNS:
        mrr (float): Averaged Reciprocal rank the entire set
        """
        no_of_queries = 0
        cum_rr = 0
        for index, query in enumerate(q_id_list):
            no_of_queries+=1
            rr = self._reciprocal_rank(query, predict_ids_list[index])
            cum_rr+=rr
        mrr = cum_rr/(no_of_queries+EPS)
        return mrr

    def _prec_at_k(self, query_id: int, predict_ids: List[int], k: int):
        """
        ARGS:
        query_id (int): id_left from queries.csv
        predict_ids (int list): list of generated predictions, sorted by confidence, with index 0 = highest confidence
        k (int): top-k predictions to look at 
        
        RETURNS:
        prec (float): number of intersecting elements between top-k documents and documents labelled 1 & 2 based on query, divided by k
        """
        golden = list(self.qrels[self.qrels['id_left']==query_id]['id_right']) # get id of article corresponding to query
        if len(golden)<k:
            return -1
        truncated_list = predict_ids[:k]
        intersect = list(set(golden).intersection(truncated_list))
        prec = len(intersect)/(k+EPS)
        return prec
    
    def _avg_prec_at_k(self, q_id_list: List[int], predict_ids_list: List[List[int]], k:int):
        """
        ARGS:
        q_id_list (int list): consolidated query ids (id_left) from queries.csv
        predict_ids_list (list (int list)): consolidated list of generated predictions
        k (int): top-k predictions to look at

        RETURNS:
        avg_prec (float): Averaged Precision@k across all predictions
        """
        no_of_queries = 0
        cum_prec = 0
        for index, query in enumerate(q_id_list):
            prec = self._prec_at_k(query, predict_ids_list[index], k)
            if prec == -1:
                print("Query {} has less than {} golden relevant documents, skipping...".format(query, k))
                continue
            no_of_queries+=1
            cum_prec+=prec
        avg_prec = cum_prec/(no_of_queries+EPS)
        return avg_prec, no_of_queries
            

    def evaluate(self, q_id_list: List[int], predict_ids_list: List[List[int]]):
        """
        Forwards the list of queries and prediction into each evaluation metric. Evaluates MRR, Hits@5, P@5. 
        ARGS:
        q_id_list (int list): consolidated id_left from queries.csv
        predict_ids_list (list (int list)): consolidated list of generated predictions
        """
        mrr = self._mean_reciprocal_rank(q_id_list, predict_ids_list)
        hits_5 = self._avg_hits_at_k(q_id_list, predict_ids_list, 5)
        prec_5, count_5 = self._avg_prec_at_k(q_id_list, predict_ids_list, 5)
        prec_10, count_10 = self._avg_prec_at_k(q_id_list, predict_ids_list, 10)
        prec_20, count_20 = self._avg_prec_at_k(q_id_list, predict_ids_list, 20)
        
        print("""MRR: {}, \t Hits @ 5: {} 
            \t Precision @ 5: {}({} Queries considered), 
            \t Precision @ 10: {}({} Queries considered),
            \t Precision @ 20: {}({} Queries considered)
        """.format(mrr, hits_5, prec_5, count_5, prec_10,count_10,prec_20,count_20))
        return
