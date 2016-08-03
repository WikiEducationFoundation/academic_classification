import mwapi
import itertools
import time


def group(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def get_text_for_revisions(revids):
    texts = {}
    chunk_size = 50
    for revid_chunk in group(chunk_size, revids):
        i = 0
        while True:
            try:
                texts.update(_try_get_text_for_revisions(revid_chunk))
                break
            except ConnectionError as e:
                if i > 5:
                    raise e
                else:
                    i += 1
                    time.sleep(5*i)
                    continue
    return texts


def _try_get_text_for_revisions(revids):
    session = mwapi.Session(
        'https://en.wikipedia.org',
        user_agent='Wiki_Ed'
    )
    results = session.get(
        action='query',
        prop='revisions',
        rvprop=['content', 'ids'],
        revids=revids
    )
    return _pull_text_from_query_results(results)


def _pull_text_from_query_results(results):
    texts = {}
    for page in results['query']['pages'].values():
        for rev in page['revisions']:
            revid = int(rev['revid'])
            if '*' in rev:
                texts[revid] = rev['*']
            else:
                texts[revid] = ''
    if 'badrevids' in results['query']:
        for revid in results['query']['badrevids']:
            revid = int(revid)
            texts[revid] = ''
    return texts
