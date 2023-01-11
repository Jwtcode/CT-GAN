
import torch
import torchvision.transforms as transforms
from .datasets import BIRDS,COCO,Flowers,CeleBA
import torchvision.datasets as datasets
class ImageDataset(object):
    def __init__(self, args, cur_img_size=None, is_distributed=False):


        self.imsize = []

        for i in range(args.STAGE):
            self.imsize.append(cur_img_size)
            cur_img_size =cur_img_size * 2

        img_size=self.imsize[args.STAGE-1]

       
        if args.dataset.lower() == 'birds':

            transform = transforms.Compose([
                transforms.Scale(int(img_size * 76 / 64)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.Scale(int(img_size * 76 / 64)),
                transforms.RandomCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
            train_dataset=BIRDS(data_dir=args.data_path, split='train',imsize=img_size,transform= transform )
            test_dataset = BIRDS(data_dir=args.data_path, split='test', imsize=img_size,transform= transform_test)
            self.n_words=train_dataset.n_words
            self.captions = train_dataset.captions
            assert train_dataset
            if is_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            else:
                train_sampler = None
                test_sampler = None

            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),

                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=test_sampler,drop_last=True)
        elif args.dataset.lower() == 'flowers':

            transform = transforms.Compose([
                transforms.Scale(int(img_size * 76 / 64)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.Scale(int(img_size * 76 / 64)),
                transforms.RandomCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
            train_dataset=Flowers(data_dir=args.data_path, split='train',imsize=img_size,transform= transform )
            #test_dataset = Flowers(data_dir=args.data_path, split='test', imsize=img_size,transform= transform_test)
            assert train_dataset
            if is_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                #test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            else:
                train_sampler = None
                #test_sampler = None

            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),

                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)
            # self.test = torch.utils.data.DataLoader(
            #    test_dataset,
            #     batch_size=10, shuffle=False,
            #     num_workers=0, pin_memory=True, sampler=test_sampler,drop_last=True)

        elif args.dataset.lower() == 'coco':
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


            train_dataset = COCO(data_dir=args.data_path, split='train', imsize=img_size, transform=transform)
            test_dataset = COCO(data_dir=args.data_path, split='test', imsize=img_size, transform=transform_test)
            assert train_dataset
            if is_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            else:
                train_sampler = None
                test_sampler=None

            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.dis_batch_size, shuffle=False,num_workers=4, pin_memory=True, sampler=test_sampler, drop_last=True)
        
        elif args.dataset.lower() == 'celeba':

            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_dataset = CeleBA(root=args.data_path, transform=transform,split='train')
            test_dataset = CeleBA(root=args.data_path, transform=transform_test,split='train')

            if is_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            else:
                train_sampler = None
                test_sampler = None
            self.train_sampler = train_sampler

            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                    num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,drop_last=True, sampler=test_sampler)
        

        elif args.dataset.lower() == 'church':
            Dt = datasets.LSUN
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_dataset = Dt(root=args.data_path, classes=["church_outdoor_train"], transform=transform)
            test_dataset = Dt(root=args.data_path, classes=["church_outdoor_val"], transform=transform_test)
            if is_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            else:
                train_sampler = None
                test_sampler = None
            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, drop_last=True, sampler=train_sampler)

            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=4, pin_memory=True,drop_last=True, sampler=test_sampler)

        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))