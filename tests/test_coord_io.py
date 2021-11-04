import transpicker.coord_io as coord_io


def display_bbox(image_path, box_file_path, save_prefix, box_width=200, is_eman_boxfile=True, is_masked=False):
    if is_eman_boxfile:
        boxes = read_eman_boxfile(box_file_path)
    else:
        boxes = read_star_file(box_file_path, box_width)
        # TODO: modify the box width input.

    img = Image.open(image_path)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = opencvImage.shape[:2]

    for box in boxes:
        xmin = int(box.x)
        ymin = img_h - (int(box.y) + box.h)
        xmax = xmin + box.w
        ymax = ymin + box.h
        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)

    if not os.path.exists("./result/split_results/pred"):
        os.makedirs('./result/split_results/pred')

    if not is_masked:
        save_name = f'/home/zhangchi/detr/result/split_results/pred/{save_prefix}_full.jpg'
        print(f'./result/split_results/pred/{save_prefix}_full.jpg has been saved.')
    else:
        save_name = f'/home/zhangchi/detr/result/split_results/pred/{save_prefix}_full_addmask.jpg'
        print(f'./result/split_results/pred/{save_prefix}_full_addmask.jpg has been saved.')
        
    cv2.imwrite(save_name, opencvImage)


def test_display():
    images_path = "/home/zhangchi/detr/coco/test/images_stitch/"
    boxs_path = "/home/zhangchi/detr/coco/test/images_stitch/saved_box_files/"
    box_width = 200
    for image in os.listdir(images_path):
        if image.endswith('.jpg'):
            image_prefix = image.split('.')[0][:-7]
            for box_file in os.listdir(boxs_path):
                if box_file.startswith(image_prefix):
                    display_bbox(images_path + image, boxs_path + box_file, image_prefix, box_width)


def test_mask_display():
    '''
    display bboxes after add post-processing.
    '''
    images_path = "/home/zhangchi/detr/coco/test/images_stitch/"
    boxs_path = "/home/zhangchi/detr/coco/test/images_stitch/saved_box_files/"
    for image in os.listdir(images_path):
        if image.endswith('.jpg'):
            image_prefix = image.split('.')[0][:-7]
            for box_file in os.listdir(boxs_path):
                if box_file.startswith(image_prefix) and box_file.endswith("mask.box"):
                    display_bbox(images_path + image, boxs_path + box_file, image_prefix, is_masked=True)


if __name__ == "__main__":
    test_display()
    test_mask_display()
    txt_path = '/home/zhangchi/cryodata/10017/'
    star_path = '/home/zhangchi/cryodata/10017/annots/'
    for txt_file in os.listdir(txt_path):
        if txt_file.endswith(".txt"):
            boxes = coord_io.read_txt_file(txt_path + txt_file, 100)
            star_file = star_path + txt_file[:-4] + '.star'
            print(star_file)
            coord_io.write_star_file(star_file, boxes)
