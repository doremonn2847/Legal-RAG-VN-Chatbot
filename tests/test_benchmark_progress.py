from benchmark_progress import render


def test_progress_render_handles_empty_results_and_escapes_status():
    page = render(dict(count=0, total=3196, status='<stopped>', eta='—',
                       summary=dict(completed_questions=0, metrics=[])))
    assert '&lt;stopped&gt;' in page
    assert '0.0% complete' in page
    assert 'nDCG@5' in page
    assert 'Latest results: 0 questions scored' in page
    assert '$table' not in page


def test_progress_render_displays_metrics():
    metric = dict(method='hybrid_bge', **{key: 0.5 for key in
        ('precision', 'recall', 'f1', 'hit_rate', 'map', 'mrr', 'ndcg', 'mean_seconds', 'p95_seconds')})
    page = render(dict(count=3196, total=3196, status='Complete', eta='Finished',
                       summary=dict(completed_questions=3196, metrics=[metric])))
    assert '100.0% complete' in page
    assert 'Hybrid + BGE' in page
    assert page.count('<td>0.5000</td>') == 9
