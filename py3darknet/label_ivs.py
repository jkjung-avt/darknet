"""label_ivs.py

Label jpg files in the input directory, in Pascal VOC (xml) format,
using a trained IEC_Traffic detector (yolov4) model.

Example usage:

$ cd ${HOME}/project/darknet
# python3 -m pip install -e .
$ python3 py3darknet/label_ivs.py \
              --cfg cfg/yolov4-ivs.cfg \
              --weights backup/yolov4-ivs_best.weights \
              --gpu 0 \
              --names data/ivs.names \
              --out_jpeg_dir ${HOME}/output/jpeg \
              --out_xml_dir  ${HOME}/output/xml  \
              --max_outputs 200 \
              ${HOME}/data/unlabeled
"""


import os
import random
import argparse
from pathlib import Path

import cv2
from py3darknet import Detector


DEFAULT_CFG = 'cfg/yolov4-jk1.cfg'
DEFAULT_WEIGHTS = 'backup/yolov4-jk1_best_0.7658.weights'
DEFAULT_NAMES = 'data/ivs.names'

FILTER_MIN_H = 20


def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(self.as_posix(), target.as_posix())


Path.copy = _copy


def parse_args():
    """Parse command line arguments."""
    desc = 'Label IEC_Traffic images'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG,
                        help='[%s]' % DEFAULT_CFG)
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help='[%s]' % DEFAULT_WEIGHTS)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (non-negative integer)')
    parser.add_argument('--names', type=str, default=DEFAULT_NAMES,
                        help='[%s]' % DEFAULT_NAMES)
    parser.add_argument('--conf', type=float, default=0.3,
                        help='confidence threshold [0.3]')
    parser.add_argument('--out_jpeg_dir', type=str, default='',
                        help='output jpeg directory []')
    parser.add_argument('--out_xml_dir', type=str, required=True,
                        help='output jpeg directory')
    parser.add_argument('--max_outputs', type=int, default=-1,
                        help='max number of output files')
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_names(names_path):
    with open(names_path, 'r') as f:
        names = [l.strip() for l in f.readlines()]
    return names


def check_detections(res):
    if any([r[2] == 0 for r in res]):    # bicycle
        return True
    elif any([r[2] == 5 for r in res]):  # truck_large
        return True
    else:
        return False


def gen_xml_txt(jpg_name, width, height, objects, clip_objs=True):
    xmls = ['<annotation>\n',
            '\t<folder>jpeg</folder>\n',
            '\t<filename>%s</filename>\n' % jpg_name,
            '\t<path></path>\n',
            '\t<source>\n',
            '\t\t<database>Unknown</database>\n',
            '\t</source>\n',
            '\t<size>\n',
            '\t\t<width>%d</width>\n' % width,
            '\t\t<height>%d</height>\n' % height,
            '\t\t<depth>3</depth>\n',
            '\t</size>\n',
            '\t<segmented>0</segmented>\n']
    for object in objects:
        cls = object['class']
        if cls == 'background':
            continue
        xmls.append('\t<object>\n')
        xmls.append('\t\t<name>%s</name>\n' % cls)
        xmin = int(object['x1'])
        ymin = int(object['y1'])
        xmax = int(object['x2'])
        ymax = int(object['y2'])
        if clip_objs:
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width-1)
            ymax = min(ymax, height-1)
        assert xmax > xmin and ymax > ymin
        xmls.extend([
            '\t\t<pose>Unspecified</pose>\n',
            '\t\t<truncated>0</truncated>\n',
            '\t\t<difficult>0</difficult>\n',
            '\t\t<bndbox>\n',
            '\t\t\t<xmin>%d</xmin>\n' % xmin,
            '\t\t\t<ymin>%d</ymin>\n' % ymin,
            '\t\t\t<xmax>%d</xmax>\n' % xmax,
            '\t\t\t<ymax>%d</ymax>\n' % ymax,
            '\t\t</bndbox>\n',
            '\t</object>\n'])
    xmls.append('</annotation>\n')
    return ''.join(xmls)


def save_xml(jpg_path, xml_path, res, names):
    """Save object labels to a XML file."""
    img = cv2.imread(jpg_path.as_posix())
    assert img is not None
    img_h, img_w, img_c = img.shape
    assert img_c == 3
    objects = []
    for det in res:
        box = det[0]
        x, y, w, h = box  # unpacking
        #conf = det[1]
        cl = int(det[2])
        assert cl < len(names)
        if h >= FILTER_MIN_H:
            objects.append({
                'class': names[cl],
                'x1': x - (w / 2),
                'y1': y - (h / 2),
                'x2': x + (w / 2),
                'y2': y + (h / 2),
            })
    txt = gen_xml_txt(jpg_path.name, img_w, img_h, objects)
    with open(xml_path.as_posix(), 'w') as f:
        f.write(txt)
    print('  wrote %s' % xml_path.as_posix())


def main():
    args = parse_args()

    src_path = Path(args.path)
    jpg_paths = list(src_path.rglob('*.jpg')) + list(src_path.rglob('*.JPG'))
    if not jpg_paths:
        raise SysetemExit('no jpg/JPG files to process!')
    random.shuffle(jpg_paths)

    jpeg_dir_path = None
    if args.out_jpeg_dir:
        os.makedirs(args.out_jpeg_dir, exist_ok=True)
        jpeg_dir_path = Path(args.out_jpeg_dir)
    os.makedirs(args.out_xml_dir, exist_ok=True)
    xml_dir_path = Path(args.out_xml_dir)

    names = get_names(args.names)

    print('Initiating the IEC_Traffic object detector...')
    detector = Detector(args.cfg, args.weights, args.gpu)

    print('Starting to process jpg images...')
    count = 0
    for jpg_path in jpg_paths:
        res = detector.detect(jpg_path.as_posix(), thresh=args.conf)
        if check_detections(res):
            count += 1
            if jpeg_dir_path:
                jpg_path.copy(jpeg_dir_path)
            xml_path = xml_dir_path / (jpg_path.stem + '.xml')
            save_xml(jpg_path, xml_path, res, names)
        if args.max_outputs > 0 and count >= args.max_outputs:
            print('Stopping at count=%d...' % count)
            break

    print('Done.')


if __name__ == '__main__':
    main()
