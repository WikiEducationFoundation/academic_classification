import argparse
import yaml
import pageclassifier as pc
import sys
import importlib
import revgather as rg
import csv
import mwparserfromhell as mwp
import logging
import itertools

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def _group(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def build_classifier(config):
    logger.info('Starting page classifier build')

    logger.info('Gathering feature extractors')
    features_extractors = []
    extractors = _gather_feature_extractors(config['feature'])
    clf = pc.PageClassifier(extractors)

    logger.info('Loading training data')
    wcode, labels = _load_training_data(config['training_file'])

    logger.info('Training page classifier')
    clf.train(wcode, labels)

    logger.info('Finished page classifier build')
    return clf


def _load_training_data(training_file):
    logger.info('Looking at training file')
    with open(training_file) as f:
        reader = csv.reader(f)
        labeled_revids = {int(r[0]): (r[1] == 'True') for r in reader}

    logger.info('Collecting page text from Wikipedia')
    texts = rg.get_text_for_revisions(labeled_revids.keys())

    logger.info('Parsing page text into Wikicode')
    labeled_wcode = [(mwp.parse(texts[key]), labeled_revids[key])
                     for key in labeled_revids]
    wcode, labels = zip(*labeled_wcode)
    return wcode, labels


def _gather_feature_extractors(features):
    features_extractors = []
    for feature in features:
        module_name, clss_name = feature['class'].rsplit('.', 1)
        module = importlib.import_module(module_name)
        clss = getattr(module, clss_name)
        features_extractors.append(clss(**feature['init_args']))
    return features_extractors


def classify(config_file, infile, outfile):
    config = yaml.load(config_file)
    clf = build_classifier(config)

    writer = csv.writer(outfile)

    logger.info('Beginning classify in batches')
    batchsize = 500
    for i, rev_batch in enumerate(_load_revids_in_batches(infile, batchsize)):
        logger.info('Get revision text for batch {0}'.format(i))
        rev_text = rg.get_text_for_revisions(rev_batch)
        logger.info('Parsing text to wikicode for batch {0}'.format(i))
        rev_wcode = [(rev_id, mwp.parse(rev_text[rev_id]))
                     for rev_id in rev_text]
        batch_revids, wcode_list = zip(*rev_wcode)
        logger.info('Classifying batch {0}'.format(i))
        batch_pred = clf.predict(wcode_list)
        writer.writerows(zip(batch_revids, batch_pred))


def _load_revids_in_batches(revfile, batchsize):
    reader = csv.reader(revfile)
    for rows in _group(batchsize, reader):
        yield [r[0] for r in rows]


def main():
    parser = argparse.ArgumentParser(
                        description="classify a page on wikipedia")
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        type=argparse.FileType('r'))
    parser.add_argument('-i',
                        '--infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o',
                        '--outfile',
                        nargs='?',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    args = parser.parse_args()
    classify(args.config, args.infile, args.outfile)


if __name__ == "__main__":
    main()
