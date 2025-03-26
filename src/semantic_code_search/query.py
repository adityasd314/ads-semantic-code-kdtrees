import gzip
import os
import pickle
import sys

import torch
from sentence_transformers import util

from semantic_code_search.embed import do_embed
from semantic_code_search.prompt import ResultScreen
from semantic_code_search.min_heap import MinHeap

def _search(query_embedding, corpus_embeddings, functions, k=5, file_extension=None):
    # TODO: filtering by file extension
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)), sorted=True)
    # out = []
    # for score, idx in zip(top_results[0], top_results[1]):
    #     out.append((score, functions[idx]))
    # print(top_results, out)
    # Create a MaxHeap to store top-k results
    min_heap = MinHeap()
    # print("MAX")
    # Insert (score, function) pairs into MaxHeap
    for idx, score in enumerate(cos_scores):
        # Insert a tuple of (score, function)
        min_heap.insert((score, functions[idx]))
        
        # If heap size exceeds k, it will automatically maintain top-k
        if len(min_heap) > k:
            min_heap.extract_min()
    out = min_heap.get_top_k(5)[::-1]
    print(out)
    return out


def _query_embeddings(model, args):
    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'r') as f:
        dataset = pickle.loads(f.read())
        if dataset.get('model_name') != args.model_name_or_path:
            print('Model name mismatch. Regenerating embeddings.')
            dataset = do_embed(args, model)
        query_embedding = model.encode(args.query_text, convert_to_tensor=True)
        results = _search(query_embedding, dataset.get(
            'embeddings'), dataset.get('functions'), k=args.n_results, file_extension=args.file_extension)
        return results


def open_in_editor(file, line, editor):
    if editor == 'vim':
        os.system('vim +{} {}'.format(line, file))
    elif editor == 'vscode':
        os.system('code --goto {}:{}'.format(file, line))


def do_query(args, model):
    if not args.query_text:
        print('provide a query')
        # todo: add a prompt here as a fallback
        sys.exit(1)

    if not os.path.isfile(args.path_to_repo + '/' + '.embeddings'):
        print('Embeddings not found in {}. Generating embeddings now.'.format(
            args.path_to_repo))
        do_embed(args, model)

    results = _query_embeddings(model, args)

    selected_idx = ResultScreen(results, args.query_text).run()
    if not selected_idx:
        sys.exit(0)  # user cancelled
    file_path_with_line = (
        results[selected_idx][1]['file'], results[selected_idx][1]['line'] + 1)
    if file_path_with_line is not None:
        open_in_editor(
            file_path_with_line[0], file_path_with_line[1], args.editor)
        sys.exit(0)
