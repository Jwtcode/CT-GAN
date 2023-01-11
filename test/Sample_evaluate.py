import os
import os.path
from PIL import Image
from torch.utils.data import Dataset


class Sample_evaluate(Dataset):
    def __init__(self,args,transform=None):

        self.transform = transform
        self.args=args
        self.data_dir = os.path.join(args.image_dir,'evaluate_images')
        self.images_names=os.listdir(self.data_dir)
        pass

    def get_imgs(self,img_path, transform=None):
        img = Image.open(img_path).convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, index):
        key = self.images_names[index]
        img_name = '%s/%s' % (self.data_dir, key)
        imgs =self.get_imgs(img_name, self.transform)
        return imgs

    def __len__(self):
        return len(self.images_names)