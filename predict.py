import os
import time

import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image

from src.MMUNet import MMUNet as UNet

import natsort

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
def show_feature_map(feature_map,save_path):
   # print(feature_map.size())
    feature_map = feature_map.squeeze(0)
   # print(feature_map.size())
    feature_map = feature_map.cpu().numpy()
    #print(feature_map.shape[0])


    save_path = os.path.join("sb", save_path)
   # cv2.imwrite("dasb.png", feature_map)
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num + 1):
        plt.subplot(int(row_num), int(row_num), index)
       # plt.imshow(feature_map[index - 1])

       # print(feature_map[index - 1].size)
        plt.axis('off')

        print(save_path)
        save_path_1=os.path.join(save_path,str(index)+".jpg")
        print(save_path)
       # imageio.imsave(save_path, feature_map[index-1])
        plt.imsave(save_path_1,feature_map[index - 1])
        #cv2.imwrite(save_path, feature_map[index-1])
        #plt.savefig('./sb/pic-{}.png'.format(str(index)))
    #plt.show()

def main():
    classes = 1  # exclude background
    weights_path = r"save_weights5/best_model.pth"
    img_path = r"/home/yhj/yhj/sbunet/GLAS/test/images"
    img_list = os.listdir(img_path)
    img_list=natsort.natsorted(img_list)
    print(img_list)
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    mean = (0.787, 0.511, 0.785)
    std = (0.157, 0.213, 0.116)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = UNet(in_channels=3, num_classes=classes+1, base_channels=64)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    time=0
    for i in range(0,len(img_list)):
        img=os.path.join(img_path,img_list[i])
        original_img = Image.open(img).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)
                                             ])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            #time.append(t_end-t_start)
            time=time+t_end-t_start
            print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
           # print(prediction.size())
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
          #  print(prediction)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
           # prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            save_path= "save6"
            save_path_1 = os.path.join(save_path, img_list[i] )
            mask.save(save_path_1)
    print("infer time",time/len(img_list))


if __name__ == '__main__':
    if not os.path.exists("./save6"):
        os.mkdir("./save6")
    main()
