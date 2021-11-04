from transpicker.read_image import image_read, image_write

if __name__ == "__main__":
    '''
    images_path = r'/home/zhangchi/detr/cryococo/10028/micrographs/'
    # images_path = r'/home/zhangchi/cryodata/empiar10093/test/'
    num = 0
    for f in os.listdir(images_path):
        num = num + 1
        # print(images_path + f)
        if os.path.isfile(images_path + f) and num < 2:
            image = image_read(images_path + f)
            if f.endswith('.mrc'):
                image_write(images_path + f[:-4] + '.jpg', image)
                # image_write(images_path + f[:-4] + '.png', image)
                imageJPG = image_read(images_path + f[:-4] + '.jpg')
                imagePNG = image_read(images_path + f[:-4] + '.png')
                print(imageJPG.shape)
                print(imageJPG.dtype)
                print(imageJPG)
                print(imagePNG.shape)
                print(imagePNG.dtype)
                print(imagePNG)

    print('num:', num)
    '''

    path = '/home/zhangchi/cryodata/parp177/micrographs/177pph_1012_3682.mrc'
    output_path = '/home/zhangchi/cryodata/parp177/justtrytry/'
    if path.endswith('.mrc'):
        image = read_mrc(path)
        print(image)
        if not np.issubdtype(image.dtype, np.float32):
            image = image.astype(np.float32)
        # image_maxmin = (image-np.min(image))/(np.max(image)-np.min(image)) * 255

        image = np.flip(image, 0)
        mean = np.mean(image)
        sd = np.std(image)
        image_norm = (image - mean) / sd
        image_norm[image_norm > 3] = 3
        image_norm[image_norm < -3] = -3
        # print(image)

        import matplotlib.pyplot as plt
        plt.imshow(image_norm, origin='lower', cmap="gray", interpolation="Hanning")
        plt.savefig(f'{output_path}177pph_1012_3682_plt.png')

        # image_write(output_path + '177pph_1012_3682.jpg', image)
        image_write(output_path + '177pph_1012_3682_norm1.jpg', image_norm)
        # image_write(output_path + '177pph_1012_3682_norm1.png', image_norm)
        # cvimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print(cvimage)
        # # im = np.array(im)
        # im = np.array(cvimage)
        # # PIL_image = Image.fromarray(cvimage)  # 这里ndarray_image为原来的numpy数组类型的输入
        # print(im)
        # # im.save('/home/zhangchi/cryodata/empiar10093mrc/test.png')
        # cv2.imwrite('/home/zhangchi/cryodata/empiar10093mrc/test2.tiff', im)