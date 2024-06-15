from PIL import Image
import numpy as np
import data.img_transforms as T
import cv2
import os

def save_heatmap_on_image(config, model, dataset):
    '''
        Generate activation heatmap for input image, according to feature maps of the hidden layers.

        input:
            config:
            model: GPU, evaluation mode (already model.eval())
            dataset:
    '''
    trainset = dataset.train
    ep_name = os.path.basename(config.MODEL.RESUME).split('.')[0]
    save_dir = os.path.join(config.OUTPUT, '_'.join(('cam', config.TEST.CAM_LAYER, ep_name)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    previous_img_list = os.listdir('/mnt/local0/zhaojiahe/results/ccreid/logs/ltcc/res50_baseline/s1/cam_layer4_checkpoint_ep55')

    print('saving heatmap...')
    for i, (img_path, pid, camid, clothid, mask) in enumerate(trainset):
        img_name = os.path.basename(img_path)

        if img_name[:-4] + '_cam_heatmap.jpg' in previous_img_list:
            # load image
            raw_img = Image.open(img_path).convert('RGB')
            
            # data transform (apply the test-time transforms)
            transform_test = T.Compose([
            T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            img = transform_test(raw_img)
            img = img.unsqueeze(0).cuda()

            # heatmap generation
            hidden_feature_maps, feature = model(img, output_hidden=True)
            heatmap = hidden_feature_maps[config.TEST.CAM_LAYER]  # [1, c, h, w]
            heatmap = heatmap.squeeze(0).mean(0).data.cpu().numpy() # [h, w]
            heatmap = cv2.resize(heatmap, (config.DATA.WIDTH, config.DATA.HEIGHT))  # resize to the same size as raw image
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # normalization to [0, 1]
            heatmap = np.uint8(heatmap * 255)

            # save heatmap alone
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            save_path = os.path.join(save_dir, img_name[:-4] + '_cam_heatmap.jpg')
            cv2.imwrite(save_path, heatmap)

            # save heatmap overlying on raw image
            raw_img = cv2.imread(img_path)
            org_img = cv2.resize(raw_img, (config.DATA.WIDTH, config.DATA.HEIGHT))
            img_with_heatmap = np.float32(org_img) + np.float32(heatmap) * 0.6
            save_path = os.path.join(save_dir, img_name[:-4] + '_cam_on_image.jpg')
            cv2.imwrite(save_path, img_with_heatmap)
    print('heatmap save finished.')